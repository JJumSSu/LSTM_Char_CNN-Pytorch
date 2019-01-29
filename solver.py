import torch
import torch.nn as nn
import numpy as np
import config
from utils import *
from model import LM
import pickle

class Solver():

    def __init__(self, train_data_path, valid_data_path, test_data_path):
       
        self.batch_size = config.batch_size # Initializing Phase
        self.embed_dim  = config.embed_dim
        self.kernels    = config.kernels
        self.channels = config.channels
        self.seq_len    = config.seq_len
        self.hidden_size  = config.hidden_size
        self.learning_rate = config.learning_rate
        self.epochs = config.epochs
        self.dic = {}
        self.word_vocab = {}
        self.char_vocab = {}
               
        self.word_vocab, self.char_vocab, self.max_len = get_vocab(train_data_path, valid_data_path, test_data_path)
        self.max_len = self.max_len + 2

        self.dic['word_vocab'] = self.word_vocab
        self.dic['char_vocab'] = self.char_vocab
        self.dic['max_len'] = self.max_len
        with open("vocab.pkl", 'wb') as f: # save dict for test
            pickle.dump(self.dic, f)
        print("dictionary saved")

        self.unique_words = len(self.word_vocab)

        self.train, self.num_batches = corpus_to_word(train_data_path, self.batch_size, True)
        self.train_idx = word_to_idx(self.train, self.word_vocab)
        self.train_idx = self.train_idx.view(self.batch_size, -1) # 20 x 46479
        
        self.train = word_to_char(self.train, self.char_vocab, self.max_len)
        self.train = torch.from_numpy(self.train) 
        self.train = self.train.view(self.batch_size, -1, self.max_len) # 20 x 46479 x 21

        self.valid, _ = corpus_to_word(valid_data_path, self.batch_size, False)
        self.valid_idx = word_to_idx(self.valid, self.word_vocab)
        self.valid_idx = self.valid_idx.view(self.batch_size, -1)

        self.valid = word_to_char(self.valid, self.char_vocab, self.max_len)
        self.valid = torch.from_numpy(self.valid)
        self.valid = self.valid.view(self.batch_size, -1, self.max_len)


    def _validate(self, seq_len, valid, valid_idx, model, h, criterion):
                
        val_loss = 0
        step = 0
        
        for j in range(0, valid.size(1)-seq_len, seq_len):

            val_input = valid[:, j : j+seq_len, :].cuda()
            val_true = valid_idx[:, (j+1) : (j+1)+seq_len].cuda().view(-1)

            y, _ = model(val_input, h)
            loss = criterion(y, val_true)
            val_loss += loss.item()
            step += 1

            model.zero_grad()
        
        print ('Validation Loss: %.3f, Perplexity: %5.2f' % (val_loss/step, np.exp(val_loss/step)))
        
        return val_loss/step


    def train_(self):


        cur_best = 10000

        model = LM(self.unique_words, self.char_vocab, self.max_len, self.embed_dim, self.channels, self.kernels, self.hidden_size)

        if torch.cuda.is_available():
            model.cuda()
        
        learning_rate = self.learning_rate
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        for epoch in range(self.epochs):

            model.train(True)

            hidden_state = [torch.zeros(2, self.batch_size, self.hidden_size).cuda()] * 2 ########

            for i in range(0, self.train.size(1)-self.seq_len, self.seq_len):

                model.zero_grad()

                inputs = self.train[:, i : i + self.seq_len,:].cuda() # 20 * 35 * 21
                targets = self.train_idx[:, (i+1) : (i+1) + self.seq_len].cuda() # 20 * 35

                temp = []           

                for state in hidden_state:
                    temp.append(state.detach())
                
                hidden_state = temp

                output, hidden_state = model(inputs, hidden_state) # initialize?
                
                loss = criterion(output, targets.view(-1))
                        
                loss.backward()
                
                nn.utils.clip_grad_norm_(model.parameters(), 5) # clipping
                
                optimizer.step()
                
                step = (i+1) // self.seq_len                    
            
                if step % 100 == 0:        
                    print ('Epoch %d/%d, Batch x Seq_Len %d/%d, Loss: %.3f, Perplexity: %5.2f' % (epoch, self.epochs, step, self.num_batches//self.seq_len, loss.item(), np.exp(loss.item())))
            
            model.eval() 
            val_loss = self._validate(self.seq_len, self.valid, self.valid_idx, model, hidden_state, criterion)
            val_perplex = np.exp(val_loss)
                        
            if cur_best-val_perplex < 1 : # pivot?
            
                if learning_rate > 0.03: 
                    learning_rate = learning_rate * 0.5
                    print("Adjusted learning_rate : %.5f"%learning_rate)
                    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
                
                else:
                    pass
            
            if val_perplex < cur_best:
                print("The current best val loss: ", val_loss)
                cur_best = val_perplex
                torch.save(model.state_dict(), 'model.pkl')

