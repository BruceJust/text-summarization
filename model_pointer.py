# -*- coding: utf-8 -*-
# @Time    : 2020/5/19 21:56
# @Author  : Bruce
# @Email   : daishaobing@outlook.com
# @File    : model.py
# @Software: PyCharm


import tensorflow as tf
import tensorflow.keras as k
import random
import numpy as np
import rouge as rouge
from tqdm import tqdm

random.seed(1234)
tf.random.set_seed(1234)

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)



class ScaleShift(k.layers.Layer):
    def __init__(self):
        super(ScaleShift, self).__init__()

    def build(self, input_shape):
        kernel_shape = (1,) * (len(input_shape) - 1) + (input_shape[-1], )
        self.log_scale = self.add_weight(name='log_scale',
                                         shape=kernel_shape,
                                         initializer='zeros')
        self.shift = self.add_weight(name='shift',
                                     shape=kernel_shape,
                                     initializer='zeros')
    def call(self, inputs, **kwargs):
        x_outs = tf.exp(self.log_scale) * inputs + self.shift
        return x_outs

def to_one_hot(x, vocab_size):
    x_mask = tf.cast(tf.greater(tf.expand_dims(x, 2), 0), 'float32')
    x = tf.one_hot(x, vocab_size)
    x = tf.reduce_sum(x_mask * x, 1, keepdims=True)
    x = tf.cast(tf.greater(x, 0.5), 'float32')
    return x

# s = ScaleShift()
# x = train_x[0:2]

# x_outs = s(x)

class Encoder(k.layers.Layer):
    def __init__(self, vocab_size, emb_dim, hid_dim):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.emb = k.layers.Embedding(vocab_size, emb_dim)
        self.drop_out = k.layers.Dropout(0.2)
        self.rnn1 = k.layers.Bidirectional(k.layers.GRU(hid_dim, return_sequences=True, return_state=True), merge_mode='sum')
        self.rnn2 = k.layers.Bidirectional(k.layers.GRU(hid_dim, return_sequences=True, return_state=True), merge_mode='sum')
        self.normal = k.layers.LayerNormalization()
        self.scale = ScaleShift()


    def call(self, inputs):
        # [batch_size, seq_len, emb_dim]
        embedded = self.emb(inputs)
        embedded = self.drop_out(embedded)
        # [batch_size, seq_len, hid_dim]
        encoder_outputs, _, _ = self.rnn1(embedded)
        encoder_outputs, hidden1, hidden2 = self.rnn1(encoder_outputs)
        encoder_outputs = self.normal(encoder_outputs)
        encoder_outputs= encoder_outputs * self.compute_mask_(inputs)

        # hidden = tf.concat([hidden1, hidden2], axis=-1)
        hidden = hidden1 + hidden2

        # [batch_size, 1, vocab_size]
        x = to_one_hot(inputs, self.vocab_size)
        x_prior = self.scale(x)
        return encoder_outputs, hidden, x_prior

    def compute_mask_(self, inputs):
        mask = tf.logical_not(tf.equal(inputs, 0))
        mask = tf.cast(mask, dtype=tf.float32)
        mask = tf.expand_dims(mask, axis=-1)
        return mask

# encoder = Encoder(20000, 256, 256)
# x = tf.ones((64, 140))
# encoder_outputs, encoder_hidden = encoder(x)




class Attention(k.layers.Layer):
    def __init__(self, hid_dim, heads):
        super(Attention, self).__init__()
        assert hid_dim % heads == 0
        self.heads = heads
        self.hid_dim = hid_dim
        self.size_per_head = int(hid_dim / heads)
        self.query = k.layers.Dense(hid_dim)
        self.key = k.layers.Dense(hid_dim)
        self.value = k.layers.Dense(hid_dim)

    def call(self, encoder_outputs, decoder_input):
        # encoder_outputs: [batch_size, seq_len_en, hid_dim]
        # decoder_outputs: [batch_size, seq_len_de, hid_dim]

        # [batch_size, seq_len, hid_dim]
        qw = self.query(decoder_input)
        kw = self.key(encoder_outputs)
        vw = self.value(encoder_outputs)

        # [batch_size, seq_len, heads, size_per_head]
        qw = tf.reshape(qw, (-1, tf.shape(qw)[1], self.heads, self.size_per_head))
        kw = tf.reshape(kw, (-1, tf.shape(kw)[1], self.heads, self.size_per_head))
        vw = tf.reshape(vw, (-1, tf.shape(vw)[1], self.heads, self.size_per_head))

        # [batch_size, heads, seq_len, size_per_head]
        qw = tf.transpose(qw, (0, 2, 1, 3))
        kw = tf.transpose(kw, (0, 2, 1, 3))
        vw = tf.transpose(vw, (0, 2, 1, 3))

        # [batch_size, heads, seq_len_q, seq_len_k]
        attention = tf.matmul(qw, kw, transpose_b=True) / self.size_per_head ** 0.5
        attention = tf.math.softmax(attention, axis=-1)

        # [batch_size, heads, seq_len_q, size_per_head]
        out = tf.matmul(attention, vw)
        # [batch_size, seq_len_q, heads, size_per_head]
        out = tf.transpose(out, (0, 2, 1, 3))
        # [batch_size, seq_len_q, hid_dim]
        out = tf.reshape(out, (-1, tf.shape(out)[1], self.hid_dim))
        return out


class Decoder(k.layers.Layer):
    def __init__(self, vocab_size, emb_dim, hid_dim, heads):
        super(Decoder, self).__init__()
        self.emb = k.layers.Embedding(vocab_size, emb_dim)
        self.rnn1 = k.layers.GRU(hid_dim, return_sequences=True, return_state=True)
        self.rnn2 = k.layers.GRU(hid_dim, return_sequences=True, return_state=True)
        # self.lstm = k.layers.Bidirectional(k.layers.GRU(hid_dim, return_sequences=True, return_state=True), merge_mode='sum')
        self.normal1 = k.layers.LayerNormalization()
        self.normal2 = k.layers.LayerNormalization()
        self.attention = Attention(hid_dim, heads)
        self.dense = k.layers.Dense(vocab_size)
        
        
    def call(self, encoder_outputs, decoder_hidden, decoder_input):
        # encoder_outputs: [batch_size, seq_len_encoder, hidden_size]
        # decoder_hidden: [batch_size, 1, hidden_size]
        # decoder_input: [batch_size, seq_len_decoder, hidden_size]
        decoder_emb = self.emb(decoder_input)
        decoder_in = self.attention(encoder_outputs, decoder_emb)
        h, hidden  = self.rnn1(decoder_in, initial_state=decoder_hidden)
        h = self.normal1(h)

        h, hidden  = self.rnn2(h, )
        h = self.normal2(h)
        # [batch_size, seq_len_decoder, vocab_size]
        output = self.dense(h)
        return output, hidden

# decoder = Decoder(20000, 256, 256, 8)
# decoder_in = tf.ones((64, 30))
# d_output = decoder(encoder_outputs, encoder_hidden, decoder_in)

class SummarizationModel(k.Model):
    def __init__(self, vocab_size, emb_dim, hid_dim, heads, dropout=0.1):
        super(SummarizationModel, self).__init__()
        self.encoder = Encoder(vocab_size, emb_dim, hid_dim)
        self.decoder = Decoder(vocab_size, emb_dim, hid_dim, heads)

    def call(self, inputs):
        x = inputs[0]
        y_input = inputs[1]
        encoder_outputs, hidden, x_prior = self.encoder(x)
        #
        output, hidden = self.decoder(encoder_outputs, hidden, y_input)
        output = (output + x_prior) / 2
        output = tf.math.softmax(output, axis=-1)
        return output


vocab_size = 6848
emb_dim = 256
hid_dim = 256
heads = 8


loss_object = k.losses.SparseCategoricalCrossentropy(
    # from_logits=True,
    reduction='none')

# @tf.function
def loss_func(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))  # 将y_true 中所有为0的找出来，标记为False
    loss_ = loss_object(y_true, y_pred)
    mask = tf.cast(mask, dtype=loss_.dtype)  # 将前面统计的是否零转换成1，0的矩阵
    loss_ *= mask     # 将正常计算的loss加上mask的权重，就剔除了padding 0的影响
    mask_count = tf.reduce_sum(mask)
    return tf.reduce_sum(loss_)  / mask_count  # 最后将loss求平均

# @tf.function
def masked_accuracy(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    y_pred = tf.math.argmax(y_pred, axis=-1)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)
    acc = tf.math.equal(y_true, y_pred)
    acc = tf.cast(acc, dtype=tf.float32)
    mask = tf.cast(mask, dtype=acc.dtype)
    mask_count = tf.reduce_sum(mask)
    acc *= mask
    acc_count = tf.reduce_sum(acc)
    return acc_count / mask_count



def build_model():
    Model = SummarizationModel(vocab_size, emb_dim, hid_dim, heads)

    Model.compile(optimizer=k.optimizers.Adam(1e-3), loss=loss_func, metrics=[masked_accuracy])
    return Model

def gen_sent(s, model, topk=3, maxlen=32):
    """beam search解码
    每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
    """
    # 输入转id
    xid = np.array([tokenizer.encode(s)] * topk)
    # 解码均以<start>开头，这里<start>的id为1
    yid = np.array([[1]] * topk)
    # 候选答案分数
    scores = [0] * topk
    # 强制要求输出不超过maxlen字
    for i in range(maxlen):
        # 直接忽略<padding>、<start>
        # model.predict: [3, 1,
        proba = model.predict([xid, yid])
        # proba = tf.math.softmax(proba, axis=-1)
        proba = proba[:, i, 2:]
        # 取对数，方便计算
        log_proba = np.log(proba + 1e-6)
        # 每一项选出topk，argsort是按小到大排序，所以topk取负数
        arg_topk = log_proba.argsort(axis=1)[:, -topk:]
        # 暂存的候选目标序列
        _yid = []
        # 暂存的候选目标序列得分
        _scores = []
        # 第一轮只需要计算topk个值
        if i == 0:
            for j in range(topk):
                # 第一个值的下标，下标包括了start
                _yid.append(list(yid[j]) + [arg_topk[0][j] + 2])
                # 第一个值的分数
                _scores.append(scores[j] + log_proba[0][arg_topk[0][j]])
        # 非第一轮需要计算topk^2个值
        else:
            # 遍历topk*topk的组合
            for j in range(topk):
                for k in range(topk):
                    _yid.append(list(yid[j]) + [arg_topk[j][k] + 2])
                    _scores.append(scores[j] + log_proba[j][arg_topk[j][k]])
            # 从中选出新的topk
            _arg_topk = np.argsort(_scores)[-topk:]
            _yid = [_yid[k] for k in _arg_topk]
            _scores = [_scores[k] for k in _arg_topk]
        yid = np.array(_yid)
        scores = np.array(_scores)
        # 输出满足条件的下标
        ends = np.where(yid[:, -1] == 2)[0]
        if len(ends) > 0:
            k = ends[scores[ends].argmax()]
            return tokenizer.decode_(yid[k][1:])
    # 如果maxlen字都找不到<end>，直接返回
    return tokenizer.decode_(yid[np.argmax(scores)][1:])

# def gen_sent(sentence, model):
#     summary = ''
#     tokens = tokenizer.encode(sentence)
#     tokens = [1] + tokens + [2]
#
#     # [1, seq_len]
#     input_tokens = tf.expand_dims(tokens, axis=0)
#     input_tokens = k.preprocessing.sequence.pad_sequences(input_tokens, padding='post', maxlen=142, value=1.)
#     start_token = tokenizer.word_to_index['<START>']
#     decode_token = tf.expand_dims([start_token], axis=0)
#     encoder_outputs, hidden, x_prior = model.encoder(input_tokens)
#     for i in range(max_len):
#         out_logits, hidden = model.decoder(encoder_outputs, hidden, decode_token)
#         out_logits = (out_logits + x_prior) / 2
#         out = tf.math.softmax(out_logits)
#         out = tf.argmax(out, axis=-1).numpy()[0]
#         if out[0] != 2:
#             new_word = tokenizer.decode(out[0])
#             summary += new_word
#             decode_token = out
#             decode_token = tf.expand_dims(decode_token, axis=0)
#         else:
#             break
#     return summary

max_len = 32
# for i in range(10):
#     sentence = train_texts[i][0]
#     summary_true = train_texts[i][1]
#     summary_pre = evaluation(sentence)
#     print('Sentence: ', sentence)
#     print('Expected Summary: ', summary_true)
#     print('New Summary: ', summary_pre)


from processing import make_data, data_generator, Tokenizer
import os


tokenizer = Tokenizer()

batch_size = 128
epochs = 50

train_x, test_x, train_y, test_y, train_text, test_text = make_data()
train_step = len(train_x) // batch_size
test_step = len(test_x) // batch_size

train_generator = data_generator(train_x, train_y, batch_size)
test_generator = data_generator(test_x, test_y, batch_size)


model = build_model()


s1 = '新华社受权于18日全文播发修改后的《中华人民共和国立法法》，修改后的立法法分为“总则”“法律”“行政法规”“地方性法规、自治条例和单行条例、规章”“适用与备案审查”“附则”等6章，共计105条。'
# 修改后的立法法全文公布
s2 = '一辆小轿车，一名女司机，竟造成9死24伤。日前，深圳市交警局对事故进行通报：从目前证据看，事故系司机超速行驶且操作不当导致。目前24名伤员已有6名治愈出院，其余正接受治疗，预计事故赔偿费或超一千万元。'
# 深圳机场9死24伤续：司机全责赔偿或超千万

class evaluate(k.callbacks.Callback):
    def __init__(self, model):
        super(evaluate, self).__init__()
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        print(gen_sent(s1, self.model))
        print(gen_sent(s2, self.model))
        # pred = []
        # summary = []
        # for item in tqdm(test_text):
        #     pred.append(gen_sent(item[0], self.model))
        #     summary.append(item[1])
        # rouge_1 = rouge.Rouge().get_scores(pred, summary)[0]['rouge-1']['f']
        # print('rouge-1:', rouge_1)




model_dir = 'model'
os.makedirs(model_dir, exist_ok=True)
file_path = os.path.join(model_dir, 'model_pointer.hdf5')

callbacks = [k.callbacks.ModelCheckpoint(file_path,
                                        monitor='val_loss',
                                        save_best_only=True,
                                        save_weights_only=True),
            evaluate(model)
             ]

history = model.fit(train_generator,
                    steps_per_epoch=train_step,
                    epochs=epochs,
                    validation_data=test_generator,
                    validation_steps=test_step,
                    callbacks=callbacks
                    )

# model.save_weights(file_path)
# model.load_weights(file_path)

