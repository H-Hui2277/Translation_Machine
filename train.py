# Copyright Shirin Yamani,2021
# Licensed under MIT licensed.
# See LICENSE.txt for more information.


import torch
import numpy as np 
import time
import matplotlib.pyplot as plt
import argparse
import torch.optim as optim
from random import shuffle
import pickle
from torchtext.data import bleu_score
from torch.nn.utils.rnn import pad_sequence

from models import *
from utilities import *

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2

class Trainer():

    def initialize_weights(self, model):
        if hasattr(model, 'weight') and model.weight.dim() > 1:
            nn.init.xavier_uniform_(model.weight.data)

    def save_dictionary(self, dictionary, input=True):
        if input is True:
            with open('saved_models/' + self.input_lang_dic.name + '2' + self.output_lang_dic.name + '/input_dic.pkl', 'wb') as f:
                pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open('saved_models/' + self.input_lang_dic.name + '2' + self.output_lang_dic.name + '/output_dic.pkl', 'wb') as f:
                pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)

    def __init__(self, lang1, lang2, data_directory, reverse, MAX_LENGTH, MAX_FILE_SIZE, batch_size, lr=0.0005, hidden_size=256, encoder_layers=3, decoder_layers=3,
                 encoder_heads=8, decoder_heads=8, encoder_ff_size=512, decoder_ff_size=512, encoder_dropout=0.1, decoder_dropout=0.1, device='cpu'):

        self.MAX_LENGTH = MAX_LENGTH
        self.MAX_FILE_SIZE = MAX_FILE_SIZE
        self.device = device

        self.input_lang_dic, self.output_lang_dic, self.input_lang_list, self.output_lang_list = load_files(lang1, lang2, data_directory, reverse, self.MAX_FILE_SIZE, self.MAX_LENGTH)
        
        for sentence in self.input_lang_list:
            self.input_lang_dic.add_sentence(sentence)

        for sentence in self.output_lang_list:
            self.output_lang_dic.add_sentence(sentence)

        self.save_dictionary(self.input_lang_dic, input=True)
        self.save_dictionary(self.output_lang_dic, input=False)

        self.tokenized_input_lang = [tokenize(sentence, self.input_lang_dic, self.MAX_LENGTH, lang=lang1 if not reverse else lang2) for sentence in self.input_lang_list]
        self.tokenized_output_lang = [tokenize(sentence, self.output_lang_dic, self.MAX_LENGTH, lang=lang2 if not reverse else lang1) for sentence in self.output_lang_list]

        self.batch_size = batch_size

        self.data_loader = load_batches(self.tokenized_input_lang, self.tokenized_output_lang, self.batch_size, self.device)

        input_size = self.input_lang_dic.n_count
        output_size = self.output_lang_dic.n_count

        #define encoder and decoder parts of transformer
        encoder_part = Encoder(input_size, hidden_size, encoder_layers, encoder_heads, encoder_ff_size, encoder_dropout, self.device)
        decoder_part = Decoder(output_size, hidden_size, decoder_layers, decoder_heads, decoder_ff_size, decoder_dropout, self.device)

        self.transformer = Transformer(encoder_part, decoder_part, self.device, PAD_TOKEN).to(self.device)
        self.transformer.apply(self.initialize_weights)

        self.loss_func = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        self.optimizer = optim.Adam(self.transformer.parameters(), lr=lr)


    def train(self, epochs, saved_model_directory):
        self.transformer.train()
        start_time = time.time()

        for epoch in range(epochs):
            #shuffle batches to prevent overfitting
            shuffle(self.data_loader)

            start_time = time.time()
            train_loss = 0

            for input, target in self.data_loader:
                #zero gradient
                self.optimizer.zero_grad()

                #pass through transformer
                output, _ = self.transformer(input, target[:,:-1])
                output_dim = output.shape[-1]

                #flatten and omit SOS from target
                output = output.contiguous().view(-1, output_dim)
                target = target[:,1:].contiguous().view(-1)

                #loss
                loss = self.loss_func(output, target)

                #backprop
                loss.backward()
                nn.utils.clip_grad_norm_(self.transformer.parameters(), 1)
                self.optimizer.step()

                train_loss += loss.item()
                
            train_loss /= len(self.data_loader)

            end_time = int(time.time() - start_time)
            torch.save(self.transformer.state_dict(), saved_model_directory + self.input_lang_dic.name +
            '2' + self.output_lang_dic.name + '/transformer_model_{}.pt'.format(epoch))

            print('Epoch: {},   Time: {}s,  Estimated {} seconds remaining.'.format(epoch, end_time, (epochs-epoch)*end_time))
            print('\tTraining Loss: {:.4f}\n'.format(train_loss))
            # self.validate()

        print('Training finished!')
        
    def validate(self, device='cuda:0'):
        self.transformer.eval()  # Set the model to evaluation mode
        all_references = []
        all_hypotheses = []

        with torch.no_grad():
            start_time = time.time()
            val_loss = 0

            for input, target in self.data_loader:
                input = input.to(device)
                target = target.to(device)

                # Generate predictions using the transformer model
                output, _ = self.transformer(input, target[:,:-1])

                # Remove the <SOS> token from the target sequence
                target = target[:, 1:]

                # Flatten the output and target tensors for computing loss
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                target = target.contiguous().view(-1)

                # Compute loss
                loss = self.loss_func(output, target)
                val_loss += loss.item()

                # Convert the output tensor back into a sequence of tokens
                output_tokens = output.argmax(dim=-1).tolist()
                target_tokens = target.tolist()

                # Pad sequences to the same length for batch processing
                output_sequences = [pad_sequence([torch.LongTensor(seq)], batch_first=True) for seq in output_tokens]
                target_sequences = [pad_sequence([torch.LongTensor(seq)], batch_first=True) for seq in target_tokens]

                # Collect references and hypotheses for BLEU score calculation
                all_references.extend([[seq.tolist()] for seq in target_sequences])
                all_hypotheses.extend([seq.tolist() for seq in output_sequences])

            val_loss /= len(self.data_loader)
            bleu = bleu_score(all_hypotheses, all_references)

            end_time = int(time.time() - start_time)
            print('Validation Time: {}s'.format(end_time))
            print('\tValidation Loss: {:.4f}'.format(val_loss))
            print('\tBLEU Score: {:.4f}\n'.format(bleu * 100))  # Multiply by 100 to get percentage

        self.transformer.train()  # Switch back to training mode if needed

    print('Validation finished!')

def main():
    parser = argparse.ArgumentParser(description='Hyperparameters for training Transformer')
    #hyperparameter loading
    parser.add_argument('--lang1', type=str, default='nederlands', help='first language in language text file')
    parser.add_argument('--lang2', type=str, default='chinese', help='second language in language text file')
    parser.add_argument('--data_directory', type=str, default='data', help='data directory')
    parser.add_argument('--reverse', type=int, default=True, help='whether to switch roles of lang1 and lang2 as input and output')

    #default hyperparameters (dont need to be inputed when called script)
    parser.add_argument('--MAX_LENGTH', type=int, default=60, help='max number of tokens in input')
    parser.add_argument('--MAX_FILE_SIZE', type=int, default=300000, help='max number of lines to read from files')
    parser.add_argument('--batch_size', type=int, default=128, help='size of batches passed through networks at each step')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate of models')
    parser.add_argument('--hidden_size', type=int, default=128, help='number of hidden layers in transformer')
    parser.add_argument('--encoder_layers', type=int, default=3, help='number of encoder layers')
    parser.add_argument('--decoder_layers', type=int, default=3, help='number of decoder layers')
    parser.add_argument('--encoder_heads', type=int, default=8, help='number of encoder heads')
    parser.add_argument('--decoder_heads', type=int, default=8, help='number of decoder heads')
    parser.add_argument('--encoder_ff_size', type=int, default=512, help='fully connected input size for encoder')
    parser.add_argument('--decoder_ff_size', type=int, default=512, help='fully connected input size for decoder')
    parser.add_argument('--encoder_dropout', type=float, default=0.1, help='dropout for encoder feed forward')
    parser.add_argument('--decoder_dropout', type=float, default=0.1, help='dropout for decoder feed forward')
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or gpu depending on availability and compatability')
    parser.add_argument('--epochs', type=int, default=100, help='number of iterations of dataset through network for training')

    parser.add_argument('--saved_model_directory', type=str, default='saved_models/', help='data directory')
    args = parser.parse_args()

    lang1 = args.lang1
    lang2 = args.lang2
    data_directory = args.data_directory
    reverse = args.reverse
    MAX_LENGTH = args.MAX_LENGTH
    MAX_FILE_SIZE = args.MAX_FILE_SIZE
    batch_size = args.batch_size
    lr = args.lr
    hidden_size = args.hidden_size
    encoder_layers = args.encoder_layers
    decoder_layers = args.decoder_layers
    encoder_heads = args.encoder_heads
    decoder_heads = args.decoder_heads
    encoder_ff_size = args.encoder_ff_size
    decoder_ff_size = args.decoder_ff_size
    encoder_dropout = args.encoder_dropout
    decoder_dropout = args.decoder_dropout
    device = args.device
    epochs = args.epochs
    saved_model_directory = args.saved_model_directory

    transformer = Trainer(lang1, lang2, data_directory, reverse, MAX_LENGTH, MAX_FILE_SIZE, batch_size, lr, hidden_size, encoder_layers, decoder_layers, 
                            encoder_heads, decoder_heads, encoder_ff_size, decoder_ff_size, encoder_dropout, decoder_dropout, device)
    transformer.train(epochs, saved_model_directory)


if __name__ == "__main__":
    main()
