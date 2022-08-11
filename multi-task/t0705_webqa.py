#! -*- coding: utf-8 -*-
# 多任务有监督预训练

from snippets import *
from bert4keras.layers import GlobalPointer

# 数据路径
filename = copy_and_shuf(u'阅读理解/抽取式/WebQA', 't0705_webqa', datatype='mrc')
corpus = jsonl_reader(filename)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        batch_masks, batch_labels = [], []
        num_no_answers = self.batch_size // 2
        for is_end, d in self.sample(random):
            if not d['answers']:
                if num_no_answers == 0:
                    continue
                else:
                    num_no_answers -= 1
            q, p = d['question'], d['passage']
            a, s = d['answers'][0] if d['answers'] else ('', -1)
            token_ids = masked_encode(q)[0]
            mask = [1] + [0] * len(token_ids[:-1])
            if s == -1:
                token_ids.extend(masked_encode(p)[0][1:])
                batch_labels.append([0, 0])
            else:
                e = s + len(a)
                pl_ids = masked_encode(p[:s])[0][1:-1]
                a_ids = masked_encode(p[s:e])[0][1:-1]
                pr_ids = masked_encode(p[e:])[0][1:]
                start = len(token_ids) + len(pl_ids)
                end = start + len(a_ids) - 1
                if start > end:
                    start, end = 0, 0
                batch_labels.append([start, end])
                token_ids.extend(pl_ids + a_ids + pr_ids)
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
                num_no_answers = self.batch_size // 2


corpus_generator = data_generator(corpus, batch_size).forfit()


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


def task_model(model):
    """完成任务所用的完整模型
    """
    masks_in = keras.layers.Input(shape=(None,))
    output = projection_layer(model.output)
    output = CustomMasking()([output, masks_in])
    output = GlobalPointer(heads=1, head_size=64, use_bias=False)(output)
    output = keras.layers.Lambda(lambda x: x[:, 0])(output)
    return keras.models.Model(model.inputs + [masks_in], output)


loss = globalpointer_crossentropy
