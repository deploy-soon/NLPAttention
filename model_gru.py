#import fire
import math
import time
import random
import pathlib
import numpy as np
from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

from misc import get_logger
from data import Data, FRENDataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):

    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        """
        :param input_dim: the size of the one-hot vectors that will be input
        :param emb_dim: the dimensionality of the embedding layer
        :param hid_dim: the dimensionality of the hidden and cell states
        :param dropout: amount of dropout to use
        """
        super().__init__()

        self.hid_dim = hid_dim

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        # src = [src len, batch size]
        embedded = self.dropout(self.embedding(src))

        # embedded = [src len, batch size, emb dim]
        outputs, hidden = self.rnn(embedded)

        # outputs = [src len, batch size, hid dim * n directions]
        return hidden

class Decoder(nn.Module):

    def __init__(self, output_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim)
        self.fc_out = nn.Linear(emb_dim + 2 * hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ipt, hidden, context):
        #input = [batch_size]
        #hidden = [n directions, batch size, hid dim]
        #context = [n directions, batch size, hid dim]
        # n directions in the decoder will both always be 1, therefore
        # hidden = [1, batch size, hid dim]
        # context = [1, batch size, hid dim]

        ipt = ipt.unsqueeze(0)
        #ipt = [1, batch size]

        embedded = self.dropout(self.embedding(ipt))
        #embedded = [1, batch size, emb dim]

        emb_con = torch.cat((embedded, context), dim = 2)
        # emb_con = [1, batch size, emb dim + hid dim)

        output, hidden = self.rnn(emb_con, hidden)

        #output = [seq len, batch size, hid dim * n directions]
        # hidden = [n directions, batch size, hid dim]

        # seq len and n directions will always be 1 in the decoder, therfore,
        # output = [1, batch size, hid dim]
        # hidden = [1, batch size, hid dim]
        
        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)),
                           dim = 1)
        # output = [batch size, emb dim + hid dim * 2]

        prediction = self.fc_out(output)
        #prediction = [batch_size, output_dim]

        return prediction, hidden


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        """
        :param src: shape (src len, batch size)
        :param trg: shape (trg len, batch size)
        :param teacher_forcing_ratio: probability to use teacher forcing
        """
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        context = self.encoder(src)
        hidden = context
        #first input to the decoder is the <sos> tokens
        ipt = trg[0, :]

        for t in range(1, trg_len):

            output, hiddenn = self.decoder(ipt, hidden, context)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)

            ipt = trg[t] if teacher_force else top1

        return outputs


class Train:

    def __init__(self, enc_emb_dim=128, dec_emb_dim=128,
                 hid_dim=256,
                 enc_dropout=0.3, dec_dropout=0.3,
                 epochs=15):
        self.logger = get_logger()
        #self.data = Data()
        self.data = FRENDataset()
        data_len = len(self.data)
        train_num = int(data_len * 0.8)
        valid_num = int(data_len * 0.1)
        test_num = data_len - train_num - valid_num
        train, valid, test = random_split(self.data, [train_num, valid_num, test_num])
        self.train_iter = DataLoader(train, batch_size = 64, shuffle=True, num_workers=4)
        self.valid_iter = DataLoader(valid, batch_size = 64, shuffle=True, num_workers=4)
        self.test_iter = DataLoader(test, batch_size = 64, shuffle=True, num_workers=4)
        #self.train_iter, self.valid_iter, self.test_iter = self.data.iterator()
        #self.input_dim = len(self.data.source.vocab)
        #self.output_dim = len(self.data.target.vocab)
        self.input_dim = self.data.source_dim
        self.output_dim = self.data.target_dim

        self.enc_emb_dim = enc_emb_dim
        self.dec_emb_dim = dec_emb_dim
        self.hid_dim = hid_dim
        self.enc_dropout = enc_dropout
        self.dec_dropout = dec_dropout

        self.encoder = Encoder(self.input_dim,
                               self.enc_emb_dim,
                               self.hid_dim,
                               self.enc_dropout)
        self.decoder = Decoder(self.output_dim,
                               self.dec_emb_dim,
                               self.hid_dim,
                               self.dec_dropout)
        self.model = Seq2Seq(self.encoder, self.decoder, device).to(device)

        self.epochs = epochs
        #target_padding_index = self.data.target.vocab.stoi[self.data.target.pad_token]
        #self.criterion = nn.CrossEntropyLoss(ignore_index = target_padding_index)
        self.criterion = nn.CrossEntropyLoss(ignore_index = self.data.end_token_pivot)

    @staticmethod
    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.normal_(param.data, mean=0, std=0.01)

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def train(self, iterator, optimizer, criterion, clip):
        self.model.train()
        epoch_loss = 0
        for i, batch in enumerate(iterator):
            src = batch[0].transpose_(0, 1).to(device)
            trg = batch[1].transpose_(0, 1).to(device)
            #src = batch.src
            #trg = batch.trg

            optimizer.zero_grad()
            output = self.model(src, trg)
            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            #trg = [(trg len -1) * batch size]
            #output = [(trg len -1) * batch size, output dim]

            loss = criterion(output, trg)
            loss.backward()

            torch.nn.utils.clip_grad_norm(self.model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(iterator)

    def evaluate(self, iterator, criterion):
        self.model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(iterator):
                src = batch[0].transpose_(0, 1).to(device)
                trg = batch[1].transpose_(0, 1).to(device)
                #src = batch.src
                #trg = batch.trg
                output = self.model(src, trg, 0.0)

                #trg = [trg len, batch size]
                #output = [trg len, batch size, output dim]

                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)
                loss = criterion(output, trg)
                epoch_loss += loss.item()
        return epoch_loss / len(iterator)

    def _epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def test(self):
        self.model.load_state_dict(torch.load(pjoin('model', 'gru.pt')))
        test_loss = self.evaluate(self.test_iter, self.criterion)
        self.logger.info(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

    def run(self):
        self.model.apply(self.init_weights)
        self.logger.info("Model trainable parametes: {}".format(self.count_parameters(self.model)))

        optimizer = optim.Adam(self.model.parameters())

        CLIP = 1
        best_valid_loss = float('inf')
        for epoch in range(self.epochs):
            start_time = time.time()
            train_loss = self.train(self.train_iter, optimizer, self.criterion, CLIP)
            valid_loss = self.evaluate(self.valid_iter, self.criterion)
            end_time = time.time()

            epoch_mins, epoch_secs = self._epoch_time(start_time, end_time)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), pjoin('model', 'gru.pt'))
            self.logger.info(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            self.logger.info(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            self.logger.info(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

if __name__ == "__main__":
    train = Train()
    train.run()
    train.test()
