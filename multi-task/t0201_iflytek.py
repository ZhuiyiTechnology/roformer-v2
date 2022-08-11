#! -*- coding: utf-8 -*-
# 多任务有监督预训练

from snippets import *

# 数据路径
filename = copy_and_shuf(u'文本分类/主题分类/IFLYTEK长文本分类', 't0201_iflytek')
corpus = jsonl_reader(filename)

# 基本参数
num_classes = next(corpus)['num_labels']
maxlen = 256


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, d in self.sample(random):
            text, label = d['text'], d['label']
            token_ids, segment_ids = masked_encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


corpus_generator = data_generator(corpus, batch_size).forfit()


def task_model(model):
    """完成任务所用的完整模型
    """
    output = pooling_layer(projection_layer(model.output))
    output = keras.layers.Dense(units=num_classes, activation='softmax')(output)
    return keras.models.Model(model.inputs, output)


loss = 'sparse_categorical_crossentropy'
