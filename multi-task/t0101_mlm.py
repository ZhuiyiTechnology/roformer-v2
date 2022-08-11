#! -*- coding: utf-8 -*-
# 多任务有监督预训练

from snippets import *
from LAC import LAC
from bert4keras.layers import Loss


def corpus():
    """语料生成器
    """
    while True:
        p = '/root/data_pretrain/wudao/WuDaoCorpus_me_shuf/*.json'
        for f in sorted(glob.glob(p)):
            with open(f, errors='ignore') as f:
                for l in f:
                    l = json.loads(l)
                    yield l['content'][:int(maxlen * 1.2)]


# 基本参数
maxlen = 512
lac = LAC(mode='seg')


def mlm_encode(text):
    """WWM语料构建
    """
    words = lac.run(text)
    rands = np.random.random(len(words))
    source, target = [tokenizer._token_start_id], [0]
    for r, w in zip(rands, words):
        ids = tokenizer.encode(w)[0][1:-1]
        if r < 0.15 * 0.8:
            source.extend([tokenizer._token_mask_id] * len(ids))
            target.extend(ids)
        elif r < 0.15 * 0.9:
            source.extend(ids)
            target.extend(ids)
        elif r < 0.15:
            source.extend(
                np.random.choice(tokenizer._vocab_size - 1, size=len(ids)) + 1
            )
            target.extend(ids)
        else:
            source.extend(ids)
            target.extend([0] * len(ids))
    source = source[:maxlen - 1] + [tokenizer._token_end_id]
    target = target[:maxlen - 1] + [0]
    return source, target


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_source_ids, batch_segment_ids, batch_target_ids = [], [], []
        for is_end, text in self.sample(random):
            source, target = mlm_encode(text)
            segment_ids = [0] * len(source)
            batch_source_ids.append(source)
            batch_segment_ids.append(segment_ids)
            batch_target_ids.append(target)
            if len(batch_source_ids) == self.batch_size or is_end:
                batch_source_ids = sequence_padding(batch_source_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_target_ids = sequence_padding(batch_target_ids)
                yield [
                    batch_source_ids, batch_segment_ids, batch_target_ids
                ], None
                batch_source_ids, batch_segment_ids, batch_target_ids = [], [], []


corpus_generator = data_generator(corpus(), batch_size).forfit()


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        # loss 计算
        loss = K.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )
        loss = K.sum(loss * y_mask) / (K.sum(y_mask) + K.epsilon())
        # loss 返回
        return loss


def task_model(model):
    """完成任务所用的完整模型
    """
    y_in = keras.layers.Input(shape=(None,))
    output = CrossEntropy(1)([y_in, model.output])
    return keras.models.Model(model.inputs + [y_in], output)


loss = None
