import numpy as np
import torch
from torch.autograd import Variable


def make_vocab(corpus, word_vocab, char_vocab, max_len):

    """ from corpus constructing word vocabulary and character dictionary """

    word_id = len(word_vocab)
    char_id = len(char_vocab) + 1
    
    for words in corpus:
        words_list = words.split()+['+'] 
        for word in words_list:
            if word not in word_vocab:
                word_vocab[word] = word_id
                word_id += 1
            for char in word:
                if char not in char_vocab:
                    char_vocab[char] = char_id
                    char_id += 1
            if max_len < len(word):
                max_len = len(word) 

    return (word_vocab, char_vocab, max_len)


def get_vocab(train_data, valid_data, test_data):

    """ construct vocabularies of given data corpuses """
    
    print("-----------------------------------------------")
    print("Constructing Vocabulary of Words and Characters")
    print("-----------------------------------------------")

    with open(train_data,'r') as f:
        train_corpus = f.readlines()
        f.close()

    with open(valid_data,'r') as f:
        valid_corpus = f.readlines()
        f.close()

    with open(test_data,'r') as f:
        test_corpus = f.readlines()
        f.close()

    word_vocab = {}
    char_vocab = {}
    max_len = 0

    word_vocab, char_vocab, max_len = make_vocab(train_corpus, word_vocab, char_vocab, max_len)
    word_vocab, char_vocab, max_len = make_vocab(valid_corpus, word_vocab, char_vocab, max_len)
    word_vocab, char_vocab, max_len = make_vocab(test_corpus, word_vocab, char_vocab, max_len)

    char_vocab['<SOT>'] = len(char_vocab)+1   
    char_vocab['<EOT>'] = len(char_vocab)+1

    print("Word Vocabulary Size : %d"%len(word_vocab))
    print("Character Vocabulary Size : %d"%len(char_vocab))
    print("Max Length of Word - 2 : %d"%max_len)

    return word_vocab, char_vocab, max_len


def corpus_to_word(data_path, batch_size):
    
    """ given text, transfrom to word lists """

    data = []

    with open(data_path,'r') as f:
        for line in f:
            words = line.split() + ['+']      
            data+= words

    total_len = len(data)
    print("Total Length of Corpus : %d"%total_len)
    
    num_batches = total_len//batch_size
    data = data[:num_batches*batch_size] # crop data

    return data, num_batches


def word_to_idx(words, word_vocab):

    """ map words to word_indexes from vocab"""

    idx = []

    for word in words:
        idx.append(word_vocab[word])
    
    return torch.from_numpy(np.asarray(idx))


def word_to_char(x, char_vocab, max_len):

    """ change given words to character embedding vectors """
    
    for i, word in enumerate(x):
        chars  = [char_vocab[c] for c in list(word)]
        chars.insert(0, char_vocab['<SOT>'])
        chars.append(char_vocab['<EOT>'])
        for _ in range(0, max_len-len(chars)):
            chars.append(0) # zero pad
        x[i] = chars # word transformed to char_embedding vectors // number of words * max_len * char_embedding_dim
    
    return np.array(x)


