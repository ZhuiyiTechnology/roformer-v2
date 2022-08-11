#! -*- coding: utf-8 -*-
# 多任务有监督预训练
# 工具代码

import os, glob, json
import numpy as np
from bert4keras.backend import K, keras, get_available_gpus
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import is_string
from bert4keras.snippets import DataGenerator
from bert4keras.snippets import sequence_padding

# 基本配置
data_center = '/root/数据中心/'
data_path = '../supervised_datasets'
pooling = 'first'
num_gpus = len(get_available_gpus())
batch_size = 32 * max(num_gpus, 1)

# 模型路径
config_path = '/root/kg/bert/chinese_roformer-v2-char_L-6_H-384_A-6/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_roformer-v2-char_L-6_H-384_A-6/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_roformer-v2-char_L-6_H-384_A-6/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

if not os.path.exists(data_path):
    os.mkdir(data_path)


def masked_encode(*args, **kwargs):
    """给encode的结果加上随机mask
    """
    token_ids, segment_ids = tokenizer.encode(*args, **kwargs)
    num_masks = int(len(token_ids[1:-1]) * 0.15) // 2 * 2
    if num_masks < 2:
        return token_ids, segment_ids
    mask_idxs = np.random.permutation(len(token_ids[1:-1]))[:num_masks]
    mask_idxs = np.sort(mask_idxs) + 1
    for i in range(0, num_masks, 2):
        k = mask_idxs[i]
        if np.random.random() < 0.8:
            token_ids[k] = tokenizer._token_mask_id
            token_ids[k + 1] = tokenizer._token_mask_id
        elif np.random.random() < 0.9:
            token_ids[k] = np.random.choice(tokenizer._vocab_size - 1) + 1
            token_ids[k + 1] = np.random.choice(tokenizer._vocab_size - 1) + 1
    return token_ids, segment_ids


def mrc_stride_split(q, p, a, s):
    """滑动窗口分割阅读理解样本
    """
    maxlen, stride = 512, 128
    e = s + len(a)
    results, n = [], 0
    max_p_len = maxlen - len(q) - 3
    while True:
        l, r = n * stride, n * stride + max_p_len
        if l <= s < e <= r:
            results.append((q, p[l:r], a, s - l))
        else:
            results.append((q, p[l:r], '', -1))
        if r >= len(p):
            return results
        n += 1


def mrc_split_and_shuf(path):
    """读取阅读理解语料，分割然后重新打乱
    """
    lines = open(path).readlines()
    fw = open(path, 'w')
    for l in lines:
        d = json.loads(l)
        if len(d['question']) > 128:
            continue
        if d['title']:
            d['passage'] = d['title'] + '\t' + d['passage']
        if not d['answers']:
            d['answers'] = [('', -1)]
        q, p = d['question'], d['passage']
        for a, s in d['answers']:
            for r in mrc_stride_split(q, p, a, s):
                if r[2].split() and r[3] > -1:
                    l = {
                        'question': r[0],
                        'passage': r[1],
                        'answers': [(r[2], r[3])]
                    }
                else:
                    l = {'question': r[0], 'passage': r[1], 'answers': []}
                fw.write(json.dumps(l, ensure_ascii=False) + '\n')
    fw.close()
    os.system('shuf %s -o %s' % (path, path))
    return path


def mcrc_stride_split(q, p, l):
    """滑动窗口分割单选型阅读理解样本
    """
    maxlen, stride = 512, 128
    results, n = [], 0
    max_p_len = maxlen - len(q) - l - 5
    while True:
        l, r = n * stride, n * stride + max_p_len
        results.append((q, p[l:r]))
        if r >= len(p):
            return results
        n += 1


def mcrc_split_and_shuf(path):
    """读取单选型阅读理解语料，分割然后重新打乱
    """
    lines = open(path).readlines()
    fw = open(path, 'w')
    for l in lines:
        d = json.loads(l)
        if len(d['question']) > 128:
            continue
        if max([len(c) for c in d['choices']]) > 128:
            continue
        if d['title']:
            d['passage'] = d['title'] + '\t' + d['passage']
        d['choices'] = [d['choices'].pop(d['answer'])] + d['choices']
        for q, p in mcrc_stride_split(
            d['question'], d['passage'], max([len(c) for c in d['choices']])
        ):
            l = {'question': q, 'passage': p, 'choices': d['choices']}
            fw.write(json.dumps(l, ensure_ascii=False) + '\n')
    fw.close()
    os.system('shuf %s -o %s' % (path, path))
    return path


def copy_and_shuf(source_path, target_path, datatype=None):
    """复制合并到指定路径并按行打乱
    """
    if os.path.exists('%s/%s.json' % (data_path, target_path)):
        return '%s/%s.json' % (data_path, target_path)

    jsons = data_center + '%s/*.json' % source_path.rstrip('/')
    os.system('cat %s | shuf > %s/%s.json' % (jsons, data_path, target_path))
    if datatype == 'mrc':
        return mrc_split_and_shuf('%s/%s.json' % (data_path, target_path))
    elif datatype == 'mcrc':
        return mcrc_split_and_shuf('%s/%s.json' % (data_path, target_path))
    else:
        return '%s/%s.json' % (data_path, target_path)


def merge_and_shuf(source_paths, target_path):
    """合并然后打乱
    """
    if os.path.exists('%s/%s.json' % (data_path, target_path)):
        return '%s/%s.json' % (data_path, target_path)

    jsons = ' '.join(source_paths)
    os.system('cat %s | shuf > %s/%s.json' % (jsons, data_path, target_path))
    return '%s/%s.json' % (data_path, target_path)


def jsonl_reader(filename, loop=True):
    """逐行读取json
    """
    while True:
        with open(filename, errors='ignore') as f:
            for l in f:
                yield json.loads(l)
        if not loop:
            break


# 预定义投影层
projection_layer = lambda x: x

# 预定义池化层
if pooling == 'first':
    pooling_layer = lambda x: keras.layers.Lambda(lambda x: x[:, 0])(x)
elif pooling == 'avg':
    pooling_layer = lambda x: keras.layers.GlobalAveragePooling1D()(x)
elif pooling == 'max':
    pooling_layer = lambda x: keras.layers.GlobalMaxPooling1D()(x)


class Adam(keras.optimizers.Optimizer):
    """重写Adam，方便参数重用
    """
    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        bias_correction=True,
        **kwargs
    ):
        super(Adam, self).__init__(**kwargs)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon or K.epislon()
        self.bias_correction = bias_correction
        self.iterations = K.variable(0, dtype='int64', name='iterations')
        self.slots = {}

    def get_slot(self, var, name):
        if (var, name) in self.slots:
            return self.slots[(var, name)]
        else:
            slot = K.zeros(K.int_shape(var), dtype=K.dtype(var))
            self.slots[(var, name)] = slot
            return slot

    @K.symbolic
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.learning_rate
        t = K.cast(self.iterations, K.floatx()) + 1
        if self.bias_correction:
            lr = lr * (
                K.sqrt(1. - K.pow(self.beta_2, t)) /
                (1. - K.pow(self.beta_1, t))
            )

        ms = [self.get_slot(p, 'm') for p in params]
        vs = [self.get_slot(p, 'v') for p in params]

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            p_t = p - lr * m / (K.sqrt(v_t) + self.epsilon)
            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            self.updates.append(K.update(p, p_t))

        return self.updates

    def get_config(self):
        config = {
            'learning_rate': self.learning_rate,
            'beta_1': self.beta_1,
            'beta_2': self.beta_2,
            'epsilon': self.epsilon,
        }
        base_config = super(Adam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SortedDict(dict):
    """按照固定的顺序返回结果
    """
    def __iter__(self):
        return iter(sorted(super(SortedDict, self).__iter__()))
