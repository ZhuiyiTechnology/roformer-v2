#! -*- coding:utf-8 -*-
# CLUE评测
# c3多项选择阅读理解
# 思路：每个选项分别与问题、篇章拼接后打分排序

import json
import numpy as np
from snippets import *
from bert4keras.backend import keras
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from bert4keras.snippets import truncate_sequences
from tqdm import tqdm

# 基本参数
num_classes = 4
maxlen = 512
batch_size = 4
epochs = 10


def load_data(filename):
    """加载数据
    格式：[(篇章, 问题, 选项, 答案id)]
    """
    D = []
    with open(filename) as f:
        data = json.load(f)
        for d in data:
            p = u'||'.join(d[0])
            for qa in d[1]:
                q = qa['question']
                while len(qa['choice']) < num_classes:
                    qa['choice'].append(u'无效答案')
                c = qa['choice'][:num_classes]
                if 'answer' in qa:
                    a = qa['choice'].index(qa['answer'])
                else:
                    a = 0
                D.append((p, q, c, a))
    return D


# 加载数据集
train_data = load_data(data_path + 'c3/m-train.json')
train_data += load_data(data_path + 'c3/d-train.json')
valid_data = load_data(data_path + 'c3/m-dev.json')
valid_data += load_data(data_path + 'c3/d-dev.json')


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (p, q, cs, a) in self.sample(random):
            for c in cs:
                p_ids = tokenizer.encode(p)[0]
                q_ids = tokenizer.encode(q)[0][1:]
                c_ids = tokenizer.encode(c)[0][1:]
                truncate_sequences(maxlen, -2, c_ids, q_ids, p_ids)
                token_ids = p_ids + q_ids + c_ids
                batch_token_ids.append(token_ids)
                batch_segment_ids.append([0] * len(token_ids))
                batch_labels.append([a])
            if len(batch_token_ids) == self.batch_size * num_classes or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)


def multichoice_crossentropy(y_true, y_pred):
    """多项选择的交叉熵
    """
    y_true = K.cast(y_true, 'int32')[::num_classes]
    y_pred = K.reshape(y_pred, (-1, num_classes))
    return K.mean(
        K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    )


def multichoice_accuracy(y_true, y_pred):
    """多项选择的准确率
    """
    y_true = K.cast(y_true, 'int32')[::num_classes, 0]
    y_pred = K.reshape(y_pred, (-1, num_classes))
    y_pred = K.cast(K.argmax(y_pred, axis=1), 'int32')
    return K.mean(K.cast(K.equal(y_true, y_pred), K.floatx()))


# 构建模型
output = base.model.output
output = keras.layers.Lambda(lambda x: x[:, 0])(output)
output = keras.layers.Dense(units=1,
                            kernel_initializer=base.initializer)(output)

model = keras.models.Model(base.model.input, output)
model.summary()

model.compile(
    loss=multichoice_crossentropy,
    optimizer=optimizer4,
    metrics=[multichoice_accuracy]
)


class Evaluator(keras.callbacks.Callback):
    """保存验证集acc最好的模型
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = self.evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('weights/c3.weights')
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )

    def evaluate(self, data):
        total, right = 0., 0.
        for x_true, y_true in data:
            y_pred = model.predict(x_true).reshape((-1, num_classes))
            y_pred = y_pred.argmax(axis=1)
            y_true = y_true[::num_classes, 0]
            total += len(y_true)
            right += (y_true == y_pred).sum()
        return right / total


def test_predict(in_file, out_file):
    """输出测试结果到文件
    结果文件可以提交到 https://www.cluebenchmarks.com 评测。
    """
    test_data = load_data(in_file)
    test_generator = data_generator(test_data, batch_size)

    results = []
    for x_true, _ in tqdm(test_generator, ncols=0):
        y_pred = model.predict(x_true).reshape((-1, num_classes))
        y_pred = y_pred.argmax(axis=1)
        results.extend(y_pred)

    fw = open(out_file, 'w')
    with open(in_file) as fr:
        data = json.load(fr)
        i = 0
        for d in data:
            for qa in d[1]:
                l = json.dumps({'id': str(qa['id']), 'label': str(results[i])})
                fw.write(l + '\n')
                i += 1
    fw.close()


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

    model.load_weights('weights/c3.weights')
    test_predict(
        in_file=data_path + 'c3/test1.0.json',
        out_file='results/c310_predict.json'
    )
    test_predict(
        in_file=data_path + 'c3/test1.1.json',
        out_file='results/c311_predict.json'
    )

else:

    model.load_weights('weights/c3.weights')
