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
        self.out_channels = config.out_channels
        self.seq_len    = config.seq_len
        self.hidden_size  = config.hidden_size
        self.learning_rate = config.learning_rate
        self.num_epochs = config.num_epochs
        self.dic = {}

        self.word_vocab = {}
        self.char_vocab = {}
        self.max_len = None
        self.data = None
        self.idx = None
        self.val_data = None
        self.val_idx = None
        self.word_vocab, self.char_vocab, self.max_len = get_vocab(train_data_path, valid_data_path, test_data_path)
        self.max_len = self.max_len + 2

        self.dic['word_vocab'] = self.word_vocab
        self.dic['char_vocab'] = self.char_vocab
        self.dic['max_len'] = self.max_len
        with open("vocab.pkl", 'wb') as f: # save dict for test
            pickle.dump(self.dic, f)
        print("dictionary saved")

        self.data, self.num_batches = corpus_to_word(train_data_path, self.batch_size)
        self.idx = word_to_idx(self.data, self.word_vocab)
        self.idx = self.idx.contiguous().view(self.batch_size, -1) # 20 x 46479
        
        self.data = word_to_char(self.data, self.char_vocab, self.max_len)
        self.data = torch.from_numpy(self.data) 
        self.data = self.data.contiguous().view(self.batch_size, -1, self.max_len) # 20 x 46479 x 21
        print(self.data.size())

        self.val_data, _ = corpus_to_word(valid_data_path, self.batch_size)
        self.val_idx = word_to_idx(self.val_data, self.word_vocab)
        self.val_idx = self.val_idx.contiguous().view(self.batch_size, -1)

        self.val_data = word_to_char(self.val_data, self.char_vocab, self.max_len)
        self.val_data = torch.from_numpy(self.val_data)
        self.val_data = self.val_data.view(self.batch_size, -1, self.max_len)


    def _validate(self, seq_len, val_data, val_label, model, h, criterion):
                
        val_loss = 0
        i = 0
        
        for j in range(0, val_data.size(1)-seq_len, seq_len):

            val_inputs = val_data[:,j:j+seq_len,:].cuda()
            val_targets = val_label[:,(j+1):(j+1)+seq_len].cuda().contiguous().view(-1)

            output, h = model(val_inputs, h)
            loss = criterion(output, val_targets)
            val_loss += loss.item()
            i += 1

            model.zero_grad()
        
        print ('Valid Loss: %.3f, Perplexity: %5.2f' %
                (val_loss/i, np.exp(val_loss/i)))
        
        return val_loss/i

    def train(self):

        best_score = 10000
        pivot = 10000

        model = LM(self.word_vocab, self.char_vocab, self.max_len, self.embed_dim, self.out_channels, self.kernels, self.hidden_size)

        if torch.cuda.is_available():
            model.cuda()
        
        learning_rate = self.learning_rate
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        for epoch in range(self.num_epochs):

            hidden_state = torch.zeros(2, self.batch_size, self.hidden_size).cuda(), torch.zeros(2,self.batch_size, self.hidden_size).cuda()

            model.train(True)

            for i in range(0, self.data.size(1)-self.seq_len, self.seq_len):

                model.zero_grad()

                inputs = self.data[:,i:i+self.seq_len,:].cuda() # 20 * 35 * 21
                targets = self.idx[:,(i+1):(i+1)+self.seq_len].cuda().contiguous() # 20 * 35
                               
                hidden_state = [state.detach() for state in hidden_state]
                
                output, hidden_state = model(inputs,hidden_state) # initialize?
                
                loss = criterion(output, targets.view(-1))
                        
                loss.backward()
                
                nn.utils.clip_grad_norm_(model.parameters(), 5) # clipping
                
                optimizer.step()
                
                step = (i+1) // self.seq_len                    
            
                if step % 100 == 0:        
                    print ('Epoch [%d/%d], Step[%d/%d], Loss: %.3f, Perplexity: %5.2f' %
                        (epoch+1, self.num_epochs, step, self.num_batches//self.seq_len,
                        loss.item(), np.exp(loss.item())))
            
            model.eval() 
            val_loss = self._validate(self.seq_len, self.val_data, self.val_idx, model, hidden_state, criterion)
            val_loss = np.exp(val_loss)
                        
            if pivot-val_loss < 0.8 : # pivot?
            
                if learning_rate > 0.03: 
                    learning_rate = learning_rate * 0.3
                    print("Adjusted learning_rate : %.5f"%learning_rate)
                    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

            pivot = val_loss

            if val_loss < best_score:
                print("The best val loss: ", val_loss)
                best_score = val_loss
                torch.save(model.state_dict(), 'model.pkl')

