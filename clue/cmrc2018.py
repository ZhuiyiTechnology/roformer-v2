#! -*- coding:utf-8 -*-
# CLUE评测
# cmrc2018阅读理解
# 思路：基于滑动窗口和GlobalPointer

import json
import numpy as np
from snippets import *
from bert4keras.backend import keras
from bert4keras.layers import GlobalPointer
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from bert4keras.snippets import lowercase_and_normalize
from tqdm import tqdm
from itertools import groupby

# 基本参数
num_classes = 119
maxlen = 512
stride = 128
batch_size = 16
epochs = 10


def stride_split(i, q, c, a, s):
    """滑动窗口分割context
    """
    # 标准转换
    q = lowercase_and_normalize(q)
    c = lowercase_and_normalize(c)
    a = lowercase_and_normalize(a)
    e = s + len(a)
    # 滑窗分割
    results, n = [], 0
    max_c_len = maxlen - len(q) - 3
    while True:
        l, r = n * stride, n * stride + max_c_len
        if l <= s < e <= r:
            results.append((i, q, c[l:r], a, s - l, e - l))
        else:
            results.append((i, q, c[l:r], '', -1, -1))
        if r >= len(c):
            return results
        n += 1


def load_data(filename):
    """加载数据
    格式：[(id, 问题, 篇章, 答案, start, end)]
    """
    D = []
    data = json.load(open(filename))['data']
    for d in data:
        for p in d['paragraphs']:
            for qa in p['qas']:
                for a in qa['answers']:
                    D.extend(
                        stride_split(
                            qa['id'], qa['question'], p['context'], a['text'],
                            a['answer_start']
                        )
                    )
                    if a['answer_start'] == -1:
                        break
    return D


# 加载数据集
train_data = load_data(data_path + 'cmrc2018/train.json')
valid_data = load_data(data_path + 'cmrc2018/dev.json')


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        batch_masks, batch_labels = [], []
        for is_end, (i, q, c, a, s, e) in self.sample(random):
            token_ids = tokenizer.encode(q)[0]
            mask = [1] + [0] * len(token_ids[:-1])
            if s == -1:
                token_ids.extend(tokenizer.encode(c)[0][1:])
                batch_labels.append([0, 0])
            else:
                cl_ids = tokenizer.encode(c[:s])[0][1:-1]
                a_ids = tokenizer.encode(c[s:e])[0][1:-1]
                cr_ids = tokenizer.encode(c[e:])[0][1:]
                start = len(token_ids) + len(cl_ids)
                end = start + len(a_ids) - 1
                batch_labels.append([start, end])
                token_ids.extend(cl_ids + a_ids + cr_ids)
            mask.extend([1] * (len(token_ids[:-1]) - len(mask)) + [0])
            batch_token_ids.append(token_ids)
            batch_segment_ids.append([0] * len(token_ids))
            batch_masks.append(mask)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_masks = sequence_padding(batch_masks)
                batch_labels = sequence_padding(batch_labels)
                yield [
                    batch_token_ids, batch_segment_ids, batch_masks
                ], batch_labels
                batch_token_ids, batch_segment_ids = [], []
                batch_masks, batch_labels = [], []


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)


class CustomMasking(keras.layers.Layer):
    """自定义mask（主要用于mask掉question部分）
    """
    def compute_mask(self, inputs, mask=None):
        return K.greater(inputs[1], 0.5)

    def call(self, inputs, mask=None):
        return inputs[0]

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def globalpointer_crossentropy(y_true, y_pred):
    """给GlobalPointer设计的交叉熵
    """
    b, l = K.shape(y_pred)[0], K.shape(y_pred)[1]
    # y_true需要重新明确一下shape和dtype
    y_true = K.reshape(y_true, (b, 2))
    y_true = K.cast(y_true, 'int32')
    y_true = y_true[:, 0] * l + y_true[:, 1]
    # 计算交叉熵
    y_pred = K.reshape(y_pred, (b, -1))
    return K.mean(
        K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    )


def globalpointer_accuracy(y_true, y_pred):
    """给GlobalPointer设计的准确率
    """
    b, l = K.shape(y_pred)[0], K.shape(y_pred)[1]
    # y_true需要重新明确一下shape和dtype
    y_true = K.reshape(y_true, (b, 2))
    y_true = K.cast(y_true, 'int32')
    y_true = y_true[:, 0] * l + y_true[:, 1]
    # 计算准确率
    y_pred = K.reshape(y_pred, (b, -1))
    y_pred = K.cast(K.argmax(y_pred, axis=1), 'int32')
    return K.mean(K.cast(K.equal(y_true, y_pred), K.floatx()))


# 构建模型
masks_in = keras.layers.Input(shape=(None,))
output = base.model.output
output = CustomMasking()([output, masks_in])
output = GlobalPointer(
    heads=1,
    head_size=base.attention_head_size,
    use_bias=False,
    kernel_initializer=base.initializer
)(output)
output = keras.layers.Lambda(lambda x: x[:, 0])(output)

model = keras.models.Model(base.model.inputs + [masks_in], output)
model.summary()

model.compile(
    loss=globalpointer_crossentropy,
    optimizer=optimizer2,
    metrics=[globalpointer_accuracy]
)


class Evaluator(keras.callbacks.Callback):
    """保存验证集acc最好的模型
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = self.evaluate(valid_data, valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('weights/cmrc2018.weights')
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )

    def evaluate(self, data, generator):
        Y_scores = np.empty((0, 1))
        Y_start_end = np.empty((0, 2), dtype=int)
        Y_true = np.empty((0, 2), dtype=int)
        for x_true, y_true in tqdm(generator, ncols=0):
            y_pred = model.predict(x_true)
            y_pred[:, 0] -= np.inf
            y_pred[:, :, 0] -= np.inf
            y_pred = y_pred.reshape((x_true[0].shape[0], -1))
            y_start_end = y_pred.argmax(axis=1)[:, None]
            y_scores = np.take_along_axis(y_pred, y_start_end, axis=1)
            y_start = y_start_end // x_true[0].shape[1]
            y_end = y_start_end % x_true[0].shape[1]
            y_start_end = np.concatenate([y_start, y_end], axis=1)
            Y_scores = np.concatenate([Y_scores, y_scores], axis=0)
            Y_start_end = np.concatenate([Y_start_end, y_start_end], axis=0)
            Y_true = np.concatenate([Y_true, y_true], axis=0)

        total, right, n = 0., 0., 0
        for k, g in groupby(data, key=lambda d: d[0]):  # 按qid分组
            g = len(list(g))
            i = Y_scores[n:n + g].argmax() + n  # 取组内最高分答案
            y_true, y_pred = Y_true[i], Y_start_end[i]
            if (y_pred == y_true).all():
                right += 1
            total += 1
            n += g

        return right / total


def test_predict(in_file, out_file):
    """输出测试结果到文件
    结果文件可以提交到 https://www.cluebenchmarks.com 评测。
    """
    test_data = load_data(in_file)
    test_generator = data_generator(test_data, batch_size)

    Y_scores = np.empty((0, 1))
    Y_start_end = np.empty((0, 2), dtype=int)
    for x_true, _ in tqdm(test_generator, ncols=0):
        y_pred = model.predict(x_true)
        y_pred[:, 0] -= np.inf
        y_pred[:, :, 0] -= np.inf
        y_pred = y_pred.reshape((x_true[0].shape[0], -1))
        y_start_end = y_pred.argmax(axis=1)[:, None]
        y_scores = np.take_along_axis(y_pred, y_start_end, axis=1)
        y_start = y_start_end // x_true[0].shape[1]
        y_end = y_start_end % x_true[0].shape[1]
        y_start_end = np.concatenate([y_start, y_end], axis=1)
        Y_scores = np.concatenate([Y_scores, y_scores], axis=0)
        Y_start_end = np.concatenate([Y_start_end, y_start_end], axis=0)

    results, n = {}, 0
    for k, g in groupby(test_data, key=lambda d: d[0]):  # 按qid分组
        g = len(list(g))
        i = Y_scores[n:n + g].argmax() + n  # 取组内最高分答案
        start, end = Y_start_end[i]
        q, c = test_data[i][1:3]
        q_tokens = tokenizer.tokenize(q)
        c_tokens = tokenizer.tokenize(c)[1:-1]
        mapping = tokenizer.rematch(c, c_tokens)  # 重匹配，直接在context取片段
        start, end = start - len(q_tokens), end - len(q_tokens)
        results[k] = c[mapping[start][0]:mapping[end][-1] + 1]
        n += g

    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

    model.load_weights('weights/cmrc2018.weights')
    test_predict(
        in_file=data_path + 'cmrc2018/test.json',
        out_file='results/cmrc2018_predict.json'
    )

else:

    model.load_weights('weights/cmrc2018.weights')
