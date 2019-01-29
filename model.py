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
       self.t_gate[0].bias.data.fill_(0)
       

    def forward(self,x):

        t = self.t_gate(x)
        h = self.h_layer(x)
        z = torch.mul(1-t,x) + torch.mul(t,h)

        return z

class LM(nn.Module):

    """   Char Aware CNN & LSTM Model   """

    def __init__(self, word_len, char_vocab, max_len, embed_dim, channels, kernels, hidden_size):

        super(LM, self).__init__()

        self.char_vocab = char_vocab # Initializing
        self.embedding = nn.Embedding(len(self.char_vocab) + 1, embed_dim, padding_idx=0) ## embedding matrix
        
        self.conv_layers = []        # Convolution
        for kernel in kernels:
            self.conv_layers.append(nn.Sequential(
                    nn.Conv2d(1, channels*kernel, kernel_size=(kernel, embed_dim)),
                    nn.Tanh(),
                    nn.MaxPool2d((max_len - kernel + 1, 1))
                    )
                )

        self.conv_layers = nn.ModuleList(self.conv_layers)

        input_size = int(channels * np.sum(kernels)) # Highway Network
        
        self.hw = HN(input_size)
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers = 2, batch_first = True, dropout=0.5) # LSTM

        self.linear = nn.Sequential(  # Prediction
                            nn.Dropout(0.5),
                            nn.Linear(hidden_size, word_len)
                        )

    def forward(self, x, h):

        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        x = x.view(-1, x.shape[2]) # 20 x 35 x 21 -> 700 x 21

        x = self.embedding(x) # embedding lookup -> 21 x 700
    
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2]) # 700 x 1 x 21 x 15
    
        y = [cnn(x).squeeze() for cnn in self.conv_layers] # CNN & highway, 525
    
        w = torch.cat(y, 1) # 700 x 525 = (1 + 2 + 3 + 4 + 5 + 7) x 25
        
        w = self.hw(w) # 700 x 525

        w = w.view(batch_size, seq_len, -1) # 20 x 35 x 525
        out, h = self.lstm(w, h) # LSTM

        out = out.contiguous().view(batch_size*seq_len,-1) # 700 x 300 -> 300 -> 10000

        out = self.linear(out) # Linear 700 x 10000

        return out,h
