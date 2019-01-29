import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


class HN(nn.Module): 

    """   Highway Network Module   """

    def __init__(self, input_size):

       super(HN, self).__init__()

       self.t_gate = nn.Sequential(nn.Linear(input_size,input_size), nn.Sigmoid())
       self.h_layer = nn.Sequential(nn.Linear(input_size,input_size), nn.ReLU())
       self.t_gate[0].bias.data.fill_(1)
       

    def forward(self,x):

        t = self.t_gate(x)
        h = self.h_layer(x)
        z = torch.mul(t,h)+torch.mul(1-t,x)

        return z


class LM(nn.Module):

    """   Char Aware CNN & LSTM Model   """

    def __init__(self, word_vocab, char_vocab, max_len, embed_dim, out_channels, kernels, hidden_size):

        super(LM, self).__init__()

        self.word_vocab = word_vocab # Initializing
        self.char_vocab = char_vocab
        self.embedding = nn.Embedding(len(char_vocab) + 1, embed_dim, padding_idx=0)
        
        self.conv_layers = []        # Convolution
        for kernel in kernels:
            self.conv_layers.append(nn.Sequential(
                    nn.Conv2d(1, out_channels*kernel, kernel_size=(kernel,embed_dim)),
                    nn.Tanh(),
                    nn.MaxPool2d((max_len-kernel+1,1))
                    )
                )

        self.conv_layers = nn.ModuleList(self.conv_layers)

        input_size = np.asscalar(out_channels * np.sum(kernels)) # Highway Network
        self.hw = HN(input_size)
        
        self.lstm = nn.LSTM(input_size, hidden_size, 2, batch_first=True, dropout=0.5) # LSTM

        self.linear = nn.Sequential(  # Prediction
                            nn.Dropout(0.5),
                            nn.Linear(hidden_size, len(word_vocab))
                        )

        #self.init_weight()


    def _init_weight(self):
        
        self.embed.weight.data.uniform_(-0.05,0.05)     # Initializing weights and biases 
        for cnn in self.conv_layers:
            cnn[0].weight.data.uniform_(-0.05,0.05)
            cnn[0].bias.data.fill_(0)       
        self.linear[1].weight.data.uniform_(-0.05,0.05)
        self.linear[1].bias.data.fill_(0)
        self.lstm.weight_hh_l0.data.uniform_(-0.05,0.05)
        self.lstm.weight_hh_l1.data.uniform_(-0.05,0.05)
        self.lstm.weight_ih_l0.data.uniform_(-0.05,0.05)
        self.lstm.weight_ih_l1.data.uniform_(-0.05,0.05)
        self.lstm.bias_hh_l0.data.fill_(1)
        self.lstm.bias_hh_l1.data.fill_(1)
        self.lstm.bias_ih_l0.data.fill_(1)
        self.lstm.bias_ih_l1.data.fill_(1)


    def forward(self, x, h):

        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        x = x.contiguous().view(-1,x.shape[2])
        
        x = self.embedding(x) # embedding lookup

        x = x.contiguous().view(x.shape[0], 1, x.shape[1], x.shape[2])
        
        y = [cnn(x).squeeze() for cnn in self.conv_layers] # CNN & highway
        w = torch.cat(y,1)
        w = self.hw(w)

        w = w.contiguous().view(batch_size,seq_len,-1)

        out, h = self.lstm(w, h) # LSTM

        out = out.contiguous().view(batch_size*seq_len,-1)

        out = self.linear(out) # Linear

        return out,h
