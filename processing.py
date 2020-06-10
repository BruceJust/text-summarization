# -*- coding: utf-8 -*-
# @Time    : 2020/5/19 23:46
# @Author  : Bruce
# @Email   : daishaobing@outlook.com
# @File    : processing.py
# @Software: PyCharm



import pandas as pd
import jieba
import tensorflow.keras as k
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random

seed=1234
tf.random.set_seed(seed)
random.seed(seed)


train_file = 'data/train_tiny.csv'


def get_data(file):
    texts = []
    df = pd.read_csv(file)
    for line in (df.values):
        text = line[0].strip()
        summary = line[1].strip()
        texts.append([text, summary])
    return texts

train_texts = get_data(train_file)

def get_vocab(texts):
    vocab = {}
    vocab['<PAD>'] = 0
    vocab['<START>'] = 1
    vocab['<END>'] = 2
    for line in texts:
        line = line[0] + line[1]
        for word in line:
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


class Tokenizer:
    def __init__(self):
        self.word_to_index = get_vocab(train_texts)
        self.index_to_word = list(self.word_to_index.keys())

    def encode(self, sentence):
        return [self.word_to_index[word] for word in sentence]

    def decode(self, token):
        return self.index_to_word[token]

    def decode_(self, tokens):
        return ''.join([self.index_to_word[i] for i in tokens])

word_to_index = get_vocab(train_texts)

def tokenize(text):
    vectors_X = []
    vectors_Y = []
    max_len_x = 0
    max_len_y = 0
    for line in text:
        vector_x = []
        vector_y = []
        for word in line[0]:
            vector_x.append(word_to_index[word])
        vector_x = vector_x
        for word in line[1]:
            vector_y.append(word_to_index[word])
        vector_y = [1] + vector_y + [2]  # START & END
        vectors_X.append(vector_x)
        vectors_Y.append(vector_y)
        max_len_x = max(max_len_x, len(vector_x))
        max_len_y = max(max_len_y, len(vector_y))
    print('max len of text:',max_len_x)
    print('max len of summary:',max_len_y)
    return vectors_X, vectors_Y, max_len_x, max_len_y


# train_x, train_y, max_len_text, max_len_summary = tokenize(train_texts)

# max_len_text = 142
# max_len_summary = 32

def make_data():
    train_x, train_y, max_len_text, max_len_summary = tokenize(train_texts)
    train_x = k.preprocessing.sequence.pad_sequences(train_x, maxlen=max_len_text, padding='post', truncating='post', value=0., dtype='int32')
    train_y = k.preprocessing.sequence.pad_sequences(train_y, maxlen=max_len_summary, padding='post', truncating='post', value=0., dtype='int32')

    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, random_state=seed, test_size=0.01)
    train_text, test_text, _, _ = train_test_split(train_texts, [0] * len(train_texts), random_state=seed, test_size=0.01)
    return train_x, test_x, train_y, test_y, train_text, test_text



def data_generator(x, y, batch_size):
    step = len(x) // batch_size
    while 1:
        n = random.randint(0, step)
        start_idx = n * batch_size
        end_idx = (n + 1) * batch_size
        yield [x[start_idx:end_idx], y[start_idx:end_idx][:, :-1]], y[start_idx:end_idx][:, 1:]


