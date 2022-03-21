#! -*- coding: utf-8 -*-
# 用GlobalPointer做中文命名实体识别
# 数据集 https://github.com/CLUEbenchmark/CLUENER2020

import json
import numpy as np
from snippets import *
from bert4keras.backend import keras
from bert4keras.backend import multilabel_categorical_crossentropy
from bert4keras.layers import EfficientGlobalPointer as GlobalPointer
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from tqdm import tqdm

maxlen = 256
epochs = 10
batch_size = 32
categories = set()


def load_data(filename):
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            d = [l['text']]
            for k, v in l.get('label', {}).items():
                categories.add(k)
                for spans in v.values():
                    for start, end in spans:
                        d.append((start, end, k))
            D.append(d)
    return D


# 标注数据
train_data = load_data(data_path + 'cluener/train.json')
valid_data = load_data(data_path + 'cluener/dev.json')
categories = list(sorted(categories))
num_classes = len(categories)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, d in self.sample(random):
            tokens = tokenizer.tokenize(d[0], maxlen=maxlen)
            mapping = tokenizer.rematch(d[0], tokens)
            start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
            token_ids = tokenizer.tokens_to_ids(tokens)
            segment_ids = [0] * len(token_ids)
            labels = np.zeros((len(categories), maxlen, maxlen))
            for start, end, label in d[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    label = categories.index(label)
                    labels[label, start, end] = 1
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels[:, :len(token_ids), :len(token_ids)])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels, seq_dims=3)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)


def globalpointer_crossentropy(y_true, y_pred):
    """给GlobalPointer设计的交叉熵
    """
    bh = K.prod(K.shape(y_pred)[:2])
    y_true = K.reshape(y_true, (bh, -1))
    y_pred = K.reshape(y_pred, (bh, -1))
    return K.mean(multilabel_categorical_crossentropy(y_true, y_pred))


def globalpointer_f1score(y_true, y_pred):
    """给GlobalPointer设计的F1
    """
    y_pred = K.cast(K.greater(y_pred, 0), K.floatx())
    return 2 * K.sum(y_true * y_pred) / K.sum(y_true + y_pred)


# 构建模型
output = base.model.output
output = GlobalPointer(
    heads=num_classes,
    head_size=base.attention_head_size,
    use_bias=False,
    kernel_initializer=base.initializer
)(output)

model = keras.models.Model(base.model.input, output)
model.summary()

model.compile(
    loss=globalpointer_crossentropy,
    optimizer=optimizer,
    metrics=[globalpointer_f1score]
)


class Evaluator(keras.callbacks.Callback):
    """保存验证集f1最好的模型
    """
    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = self.evaluate(valid_generator)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('weights/cluener.weights')
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )

    def evaluate(self, data):
        X, Y, Z = 1e-10, 1e-10, 1e-10
        for x_true, y_true in data:
            y_pred = (model.predict(x_true) > 0).astype(int)
            X += (y_pred * y_true).sum()
            Y += y_pred.sum()
            Z += y_true.sum()
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        return f1, precision, recall


def test_predict(in_file, out_file):
    """输出测试结果到文件
    结果文件可以提交到 https://www.cluebenchmarks.com 评测。
    """
    test_data = load_data(in_file)
    test_generator = data_generator(test_data, batch_size)

    results = []
    for x_true, _ in tqdm(test_generator, ncols=0):
        y_pred = model.predict(x_true)
        for y in y_pred:
            results.append(np.where(y > 0))

    fw = open(out_file, 'w', encoding='utf-8')
    with open(in_file) as fr:
        for l, r in zip(fr, results):
            l = json.loads(l)
            l['label'] = {}
            tokens = tokenizer.tokenize(l['text'], maxlen=maxlen)
            mapping = tokenizer.rematch(l['text'], tokens)
            for label, start, end in zip(*r):
                label = categories[label]
                start, end = mapping[start][0], mapping[end][-1]
                if label not in l['label']:
                    l['label'][label] = {}
                entity = l['text'][start:end + 1]
                if entity not in l['label'][label]:
                    l['label'][label][entity] = []
                l['label'][label][entity].append([start, end])
            l = json.dumps(l, ensure_ascii=False)
            fw.write(l + '\n')
    fw.close()


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

    model.load_weights('weights/cluener.weights')
    test_predict(
        in_file=data_path + 'cluener/test.json',
        out_file='results/cluener_predict.json'
    )

else:

    model.load_weights('weights/cluener.weights')
