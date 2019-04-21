# chapter8 seq2seq

> takes a sequence as input and produces another sequence

常常用于翻译之中。

## Capturing More from a Sequence: Bidirectional Recurrent Models


从左到右捕捉到的信息与从右到左是不一样的。

## Capturing More from a Sequence: Attention


# 如何评估

在分类中，我们一般使用精度、准确度、召回率和F1来进行评估分类器的性能。
但是在生成模型中，无法使用。比如一个法语可以有多种英语翻译。我们使用一个参考输出。
一般使用两种评估

1. human evaluation 

* 使用打分投票

2. automatic evaluation

* n-gram overlap–based metrics(BLEU, ROUGE, and METEOR)

```python
from nltk.translate.bleu_score import sentence_bleu
reference = [['The', 'cat', 'is', 'on', 'the', 'mat']]
candidate = ['The', 'cat', 'sat', 'on', 'the', 'mat']
score = sentence_bleu(reference, candidate)
```
[BLEU](https://www.jianshu.com/p/15c22fadcba5)

就是每一个n-gram的加权求和求BP

* perplexity


# Example: Neural Machine Translation

1. 双向RNN捕捉信息
2. 使用注意力机制来加权信息

## 2019年4月20日 理解 PackedSequence机制

原因在于每个句子的长度不一样，无法batch放入到神经网络里面训练，使用PackedSequence来进行打包

```
a = [torch.tensor([1,2,3]), torch.tensor([3,4])]
b = torch.nn.utils.rnn.pad_sequence(a, batch_first=True)
>>>>
tensor([[ 1,  2,  3],
[ 3,  4,  0]])
torch.nn.utils.rnn.pack_padded_sequence(b, batch_first=True, lengths=[3,2]
>>>>PackedSequence(data=tensor([ 1,  3,  2,  4,  3]), batch_sizes=tensor([ 2,  2,  1]))
```

当我们输入是[1,2,3]以及[3,4]的时候，无法batch放入到nn里面，可以使用打包，比如神经网络每次batch输入都不一样

[PackedSequence](https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch)

比如第一个时间段输入2个，第二个时间段也是2个，第三个时间段1个。

代码在[NMT](https://github.com/joosthub/PyTorchNLPBook/blob/master/chapters/chapter_8/8_5_NMT/8_5_NMT_scheduled_sampling.ipynb)




