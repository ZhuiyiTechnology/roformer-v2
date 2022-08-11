# RoFormerV2

RoFormer升级版，主要通过结构的简化来提升速度，并通过无监督预训练和有监督预训练的结合来提升效果，从而达到了速度与效果的“双赢”。

<img src="https://kexue.fm/usr/uploads/2022/03/1268810640.png" width=840>

## 介绍

- 博客：https://kexue.fm/archives/8998

## 环境

bert4keras >= 0.11.0

## 下载

- **Small版**：[chinese_roformer-v2-char_L-6_H-384_A-6.zip](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roformer-v2-char_L-6_H-384_A-6.zip)、[百度云](https://pan.baidu.com/s/1huUrC9P60Afggo8AfiUcmA)(提取码：ttn4)
- **Base版**：[chinese_roformer-v2-char_L-12_H-768_A-12.zip](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roformer-v2-char_L-12_H-768_A-12.zip)、[百度云](https://pan.baidu.com/s/1qcnN4LVKVe0-mnHlkN3-6Q)(提取码：pfoh)
- **Large版**：[chinese_roformer-v2-char_L-24_H-1024_A-16.zip](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roformer-v2-char_L-24_H-1024_A-16.zip)、[百度云](https://pan.baidu.com/s/1QiJWSZrGxn8vek-8myvL6w)(提取码：npfv)

## 训练

多任务训练代码参考 https://github.com/ZhuiyiTechnology/roformer-v2/tree/main/multi-task

## 配置

- **Small版**：两张3090（24G），先用无监督MLM训练了100万步（maxlen为512），然后有监督多任务训练了75万步（maxlen从64到512不等，取决于任务），batch_size为512，优化器为LAMB；

- **Base版**：四张3090（24G），先用无监督MLM训练了100万步（maxlen为512），然后有监督多任务训练了75万步（maxlen从64到512不等，取决于任务），batch_size为512，优化器为LAMB；

- **Large版**：两张A100（80G），先用无监督MLM训练了100万步（maxlen为512），然后有监督多任务训练了50万步（maxlen从64到512不等，取决于任务），batch_size为512，优化器为LAMB。

注：无监督的训练数据为280G，有监督的训练数据约为20G（77个标注数据集，构建了92个任务进行多任务训练，涵盖文本分类、文本匹配、阅读理解、信息抽取、指代消解等常见自然语言理解任务），large版的有监督训练步数更少，是因为20G的标注数据实在不够“喂饱”large级别的模型，继续训练下去出现了过拟合现象。

## 引用

Bibtex：

```tex
@techreport{roformerv2,
  title={RoFormerV2: A Faster and Better RoFormer - ZhuiyiAI},
  author={Jianlin Su, Shengfeng Pan, Bo Wen, Yunfeng Liu},
  year={2022},
  url="https://github.com/ZhuiyiTechnology/roformer-v2",
}
```

## 联系

- 邮箱：ai@wezhuiyi.com
- 追一科技：https://zhuiyi.ai
