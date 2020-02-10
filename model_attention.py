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
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader

from misc import get_logger
from data import Data, FRENDataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):

    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        """
        :param input_dim: the size of the one-hot vectors that will be input
        :param emb_dim: the dimensionality of the embedding layer
        :param enc_hid_dim: the dimensionality of the encoder hidden states
        :param dec_hid_dim: the dimensionality of the decoder hidden states
        :param dropout: amount of dropout to use
        """
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        # src = [src len, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src len, batch size, emb dim]

        outputs, hidden = self.rnn(embedded)
        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # hidden is stacked [forward1, backward1, forward2, backward2, ...]
        # just use last forwawd and backward context
        # hidden [-2, :, :] is the last forwards RNN
        # hidden [-1, :, :] is the last backwards RNN

        # use final hidden state of the forwards and backwards to put in 
        # hidden state of the decoder
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        # outputs = [ src len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]

        return outputs, hidden


class Attention(nn.Module):

    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim * 2 + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)

    def forward(self, hidden, encoder_outputs):
        """
        :param hidden: [batch size, dec hid dim]
        :param encoder_outputs: [src len, batch size, enc hid dim*2]
        merge hidden states of decoder and bidrectional output of encoder
        """
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        #attention = [batch size, src len]

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):

    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim,
                 dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + enc_hid_dim * 2, dec_hid_dim)
        self.fc_out = nn.Linear(emb_dim + 2 * enc_hid_dim + dec_hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ipt, hidden, encoder_outputs):
        #input = [batch_size]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]

        ipt = ipt.unsqueeze(0)
        #ipt = [1, batch size]

        embedded = self.dropout(self.embedding(ipt))
        #embedded = [1, batch size, emb dim]

        attn = self.attention(hidden, encoder_outputs)
        attn = attn.unsqueeze(1)
        # attn = [batch size, 1, src len]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        #encoder_outputs = [batch size, sec len, enc hid dim * 2]

        weighted = torch.bmm(attn, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        #weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim = 2)
        #rnn_input = [1, batch size, enc hid dim * 2 + emb dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        #output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n directions, batch size, dec hid dim]

        # seq len and n directions will always be 1 in the decoder, therfore,
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        #prediction = [batch_size, output_dim]

        return prediction, hidden.squeeze(0)


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

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

        encoder_outputs, hidden = self.encoder(src)
        #first input to the decoder is the <sos> tokens
        ipt = trg[0, :]

        for t in range(1, trg_len):

            output, hidden = self.decoder(ipt, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)

            ipt = trg[t] if teacher_force else top1

        return outputs


class Train:

    def __init__(self, enc_emb_dim=128, dec_emb_dim=128,
                 enc_hid_dim=256, dec_hid_dim=256,
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
        self.train_iter = DataLoader(train, batch_size = 128, shuffle=True, num_workers=4)
        self.valid_iter = DataLoader(valid, batch_size = 128, shuffle=True, num_workers=4)
        self.test_iter = DataLoader(test, batch_size = 128, shuffle=True, num_workers=4)
        #self.train_iter, self.valid_iter, self.test_iter = self.data.iterator()
        #self.input_dim = len(self.data.source.vocab)
        #self.output_dim = len(self.data.target.vocab)
        self.input_dim = self.data.source_dim
        self.output_dim = self.data.target_dim

        self.enc_emb_dim = enc_emb_dim
        self.dec_emb_dim = dec_emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.enc_dropout = enc_dropout
        self.dec_dropout = dec_dropout

        self.encoder = Encoder(self.input_dim,
                               self.enc_emb_dim,
                               self.enc_hid_dim,
                               self.dec_hid_dim,
                               self.enc_dropout)
        self.attention = Attention(self.enc_hid_dim, self.dec_hid_dim)
        self.decoder = Decoder(self.output_dim,
                               self.dec_emb_dim,
                               self.enc_hid_dim,
                               self.dec_hid_dim,
                               self.dec_dropout,
                               self.attention)
        self.model = Seq2Seq(self.encoder, self.decoder, device).to(device)

        self.epochs = epochs
        #target_padding_index = self.data.target.vocab.stoi[self.data.target.pad_token]
        #self.criterion = nn.CrossEntropyLoss(ignore_index = target_padding_index)
        self.criterion = nn.CrossEntropyLoss(ignore_index = self.data.end_token_pivot)

    @staticmethod
    def init_weights(m):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)

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
        print(self.model)
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
