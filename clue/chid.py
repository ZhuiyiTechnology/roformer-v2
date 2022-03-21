#! -*- coding:utf-8 -*-
# CLUE评测
# chid成语阅读理解（多项选择）
# 思路：每个选项分别与问题、篇章拼接后打分排序

import json, re
import numpy as np
from snippets import *
from bert4keras.backend import keras
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from scipy.optimize import linear_sum_assignment
from itertools import groupby
from tqdm import tqdm

# 基本参数
num_classes = 10
maxlen = 64
batch_size = 12
epochs = 5


def load_data(q_file, a_file=None):
    """加载数据
    格式：[(左文本, 右文本, 答案id, 候选词集)]
    """
    D = []
    with open(q_file) as fq:
        if a_file is not None:
            A = json.load(open(a_file))
        for i, l in enumerate(fq):
            l = json.loads(l)
            assert len(l['candidates']) == num_classes
            for c in l['content']:
                for k in re.finditer('#idiom\d{6}#', c):
                    cl = re.sub('#idiom\d{6}#', '_____', c[:k.start()])
                    cr = re.sub('#idiom\d{6}#', '_____', c[k.end():])
                    while len(cl) + len(cr) > maxlen - 8:
                        if len(cl) >= len(cr):
                            cl = cl[1:]
                        else:
                            cr = cr[:-1]
                    k = c[k.start():k.end()]
                    a = (i, k, 0 if a_file is None else A[k])
                    D.append((cl, cr, a, l['candidates']))
    return D


# 加载数据集
train_data = load_data(
    data_path + 'chid/train.json', data_path + 'chid/train_answer.json'
)
valid_data = load_data(
    data_path + 'chid/dev.json', data_path + 'chid/dev_answer.json'
)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (cl, cr, (_, _, a), cs) in self.sample(random):
            token_ids = tokenizer.encode(cl, cr)[0]
            k = token_ids.index(tokenizer._token_end_id) + 1
            for c in cs:
                ids = tokenizer.encode(c)[0][1:]
                batch_token_ids.append(token_ids[:k] + ids + token_ids[k:])
                batch_segment_ids.append([0] * len(batch_token_ids[-1]))
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
    optimizer=optimizer2,
    metrics=[multichoice_accuracy]
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
            model.save_weights('weights/chid.weights')
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )

    def evaluate(self, data, generator):
        total, right = 0, 0.
        logits = np.empty((0, num_classes))
        for x_true, y_true in tqdm(generator, ncols=0):
            y_pred = model.predict(x_true).reshape((-1, num_classes))
            logits = np.concatenate([logits, y_pred], axis=0)
        for _, g in groupby(data, key=lambda d: d[2][0]):
            y_true = np.array([d[2][2] for d in g])
            costs = -logits[total:total + len(y_true)]
            y_pred = linear_sum_assignment(costs)[1]
            total += len(y_true)
            right += (y_true == y_pred).sum()
        return right / total


def test_predict(in_file, out_file):
    """输出测试结果到文件
    结果文件可以提交到 https://www.cluebenchmarks.com 评测。
    """
    test_data = load_data(in_file)
    test_generator = data_generator(test_data, batch_size)

    logits = np.empty((0, num_classes))
    for x_true, _ in tqdm(test_generator, ncols=0):
        y_pred = model.predict(x_true).reshape((-1, num_classes))
        logits = np.concatenate([logits, y_pred], axis=0)

    results, total = {}, 0
    for _, g in groupby(test_data, key=lambda d: d[2][0]):
        keys = [d[2][1] for d in g]
        costs = -logits[total:total + len(keys)]
        y_pred = linear_sum_assignment(costs)[1]
        for k, r in zip(keys, y_pred):
            results[k] = int(r)
        total += len(keys)

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

    model.load_weights('weights/chid.weights')
    test_predict(
        in_file=data_path + 'chid/test1.0.json',
        out_file='results/chid10_predict.json'
    )
    test_predict(
        in_file=data_path + 'chid/test1.1.json',
        out_file='results/chid11_predict.json'
    )

else:

    model.load_weights('weights/chid.weights')
