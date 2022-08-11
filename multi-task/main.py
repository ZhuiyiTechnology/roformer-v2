#! -*- coding: utf-8 -*-
# 多任务有监督预训练
# 训练程序

from snippets import *
from bert4keras.models import build_transformer_model
from bert4keras.models import data_parallel
from bert4keras.optimizers import extend_with_weight_decay
from bert4keras.optimizers import extend_with_layer_adaptation
from bert4keras.optimizers import extend_with_piecewise_linear_lr
from bert4keras.optimizers import extend_with_gradient_accumulation
from bert4keras.snippets import parallel_apply_generator
from keras.utils import Progbar
import t0101_mlm
import t0201_iflytek
import t0202_tnews
import t0203_thucnews
# import xxxx
import t0705_webqa
# import yyyy

# 任务列表
tasks = [
    (t0101_mlm, 60),
    (t0201_iflytek, 2),
    (t0202_tnews, 5),
    (t0203_thucnews, 4),
    # (xxxx, xx),
    (t0203_thucnews, 4),
    # (yyyy, yy),
]

# 任务配置
freqs = [f for t, f in tasks]
tasks = [t for t, f in tasks]
steps_per_epoch = 1000
epochs = 10000

# 预训练模型
base = build_transformer_model(
    config_path,
    checkpoint_path=None,
    model='roformer_v2',
    with_mlm='linear',
    return_keras_model=False
)
base.model.load_weights('../v3c7s/roformer.v2.4000.weights')
last_layer = 'Transformer-%s-FeedForward-Norm' % (base.num_hidden_layers - 1)
output = base.model.get_layer(last_layer).output
encoder = keras.models.Model(base.model.inputs, output)
encoder = data_parallel(encoder)
mlm_output = base.apply_final_layers(encoder.output)
mlm_model = keras.models.Model(encoder.inputs, mlm_output)

# 构建优化器
AdamW = extend_with_weight_decay(Adam, name='AdamW')
LAMB = extend_with_layer_adaptation(AdamW, name='LAMB')
LAMBLR = extend_with_piecewise_linear_lr(LAMB, name='LAMBLR')
LAMBLRG = extend_with_gradient_accumulation(LAMBLR, name='LAMBLRG')
optimizer = LAMBLRG(
    learning_rate=1.76e-4,
    bias_correction=False,
    weight_decay_rate=0.01,
    grad_accum_steps=8,
    lr_schedule={
        32000: 1,
        320000: 0.5,
        1280000: 0.1,
        2560000: 0.01
    }
)

# 子任务模型
task_models = [getattr(t, 'task_model')(encoder) for t in tasks[1:]]
task_models = [t0101_mlm.task_model(mlm_model)] + task_models
for i, model in enumerate(task_models):
    model.compile(loss=getattr(tasks[i], 'loss'), optimizer=optimizer)


def parallel_train_data(workers=1):
    """通过多进程，使得数据生成与模型训练并行
    说明：这里最关键的地方是每个generator只能放在固定的进程中，
          不可以多个进程读取同一个generator，加锁都不行。
    """
    def split_tasks(freqs, workers=2):
        """将任务尽量地按频率均匀分配
	"""
        partitions, start = [], 0
        while workers > 0:
            cum_freqs = np.cumsum(freqs[start:])
            span = 1. * cum_freqs[-1] / workers
            end = np.where(cum_freqs >= span)[0][0] + 1
            end = start + min(end, len(cum_freqs) - workers + 1)
            partitions.append([start, end])
            start = end
            workers -= 1
        return partitions

    def task_ids(freqs, offset=0):
        """随机返回任务id
        """
        probs = 1. * np.array(freqs) / sum(freqs)
        while True:
            yield np.random.choice(len(probs), p=probs) + offset

    def task_batch(i):
        """返回一个batch的任务数据
        """
        return i, next(getattr(tasks[i], 'corpus_generator'))

    generators, weights = [], []
    for part in split_tasks(freqs, workers):
        generators.append(
            parallel_apply_generator(
                func=task_batch,
                iterable=task_ids(freqs[part[0]:part[1]], part[0]),
                workers=1,
                max_queue_size=128
            )
        )
        weights.append(sum(freqs[part[0]:part[1]]))
    probs = 1. * np.array(weights) / sum(weights)

    while True:
        yield next(np.random.choice(generators, p=probs))


# ============ 训练代码 ============

losses = open('train.losses', 'w')  # 记录收敛过程
train_data = parallel_train_data(8)  # 非阻塞生成

for epoch in range(epochs):
    print('Epoch %s/%s' % (epoch + 1, epochs))
    pbar = Progbar(steps_per_epoch)
    pbar._values = SortedDict()
    for step in range(steps_per_epoch):
        _, (i, batch) = next(train_data)
        loss = task_models[i].train_on_batch(*batch)
        if isinstance(loss, list):
            loss = loss[0]
        line = [tasks[i].__name__, float(loss)]
        losses.write(json.dumps(line) + '\n')
        losses.flush()
        pbar.update(step + 1, values=[(tasks[i].__name__, loss)])
    base.model.save_weights('roformer.v2.mt.weights')
    if (epoch + 1) % 100 == 0:
        base.model.save_weights('roformer.v2.mt.%s.weights' % (epoch + 1))
    print()

losses.close()
