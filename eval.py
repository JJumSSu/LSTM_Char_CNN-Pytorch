import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import pickle
from utils import *
from model import HN,LM
import config
import numpy as np
import argparse
  
def eval(seq_len, val_data, val_idx, model, h, criterion):
    
    model.eval()
    val_loss = 0
    count = 0
    for j in range(0,val_data.size(1)-seq_len,seq_len):
        
        val_inputs = Variable(val_data[:,j:j+seq_len,:].cuda(), volatile = True)
        val_targets = Variable(val_idx[:,(j+1):(j+1)+seq_len].cuda(), volatile = True)

        h = [state.detach() for state in h]

        output,h = model(val_inputs,h)
        loss = criterion(output,val_targets.view(-1))
        val_loss += loss.item()
        count += 1
        model.zero_grad()
        
    print ('Test  Loss: %.3f, Perplexity: %5.2f' %
            (val_loss/count, np.exp(val_loss/count)))
    
    return val_loss/count


def main(test_data_path):
    
    dic = pickle.load(open('vocab.pkl','rb'))
    word_vocab = dic['word_vocab']
    char_vocab = dic['char_vocab']
    max_len = dic['max_len']
    batch_size = config.batch_size
    embed_dim = config.embed_dim
    out_channels = config.out_channels
    kernels = config.kernels
    hidden_size = config.hidden_size
    learning_rate = config.learning_rate
    seq_len = config.seq_len

    test_data, _ = corpus_to_word(test_data_path, batch_size)
    
    test_idx = word_to_idx(test_data,word_vocab)
    test_idx = test_idx.contiguous().view(batch_size, -1)

    test_data = word_to_char(test_data, char_vocab, max_len)
    test_data = torch.from_numpy(test_data)
    test_data = test_data.contiguous().view(batch_size, -1, max_len)

    model = LM(word_vocab,char_vocab,max_len,embed_dim,out_channels,kernels,hidden_size)

    if torch.cuda.is_available():
        model.cuda()


    model.load_state_dict(torch.load('model.pkl'))

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5,patience=1,verbose=True)

    hidden_state = (Variable(torch.zeros(2,batch_size,hidden_size).cuda(), volatile=False), 
                    Variable(torch.zeros(2,batch_size,hidden_size).cuda(), volatile=False))
    model.eval()
    test_loss = eval(seq_len,test_data,test_idx,model,hidden_state, criterion)
    test_loss = np.exp(test_loss)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-test", dest="test", default = "test.txt")
    args = parser.parse_args()

    main(args.test)
            
                  







