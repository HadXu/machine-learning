# 第一章

> 自然语言理解是计算机解决人类问题的终极目标。

## 数据的表达方式

* one-hot

```
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
 
corpus = ['Time flies flies like an arrow.',
          'Fruit flies like a banana.']
one_hot_vectorizer = CountVectorizer(binary=True)
one_hot = one_hot_vectorizer.fit_transform(corpus).toarray()
sns.heatmap(one_hot, annot=True,
            cbar=False, xticklabels=vocab,
            yticklabels=['Sentence 2'])
```

* TF-IDF

```
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
 
tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus).toarray()
sns.heatmap(tfidf, annot=True, cbar=False, xticklabels=vocab,
            yticklabels= ['Sentence 1', 'Sentence 2'])
```

## 标量、向量、张量

个人理解,标量就是一个数,向量就是一堆数，而张量就是一堆向量，从维度上看标量为0维，向量为1维，张量为2维及以上。

## Pytorch入门

Pytorch是我最喜欢的一个深度学习框架，如果你对Numpy熟悉的话，使用Pytorch就非常简单，在个人看来，Pytorch就是GPU版的Numpy。


## exercise


1. Create a 2D tensor and then add a dimension of size 1 inserted at dimension 0.

2. Remove the extra dimension you just added to the previous tensor.

3. Create a random tensor of shape 5x3 in the interval [3, 7)

4. Create a tensor with values from a normal distribution (mean=0, std=1).

5. Retrieve the indexes of all the nonzero elements in the tensor torch.Tensor([1, 1, 1, 0, 1]).

6. Create a random tensor of size (3,1) and then horizontally stack four copies together.

7. Return the batch matrix-matrix product of two three-dimensional matrices (a=torch.rand(3,4,5), b=torch.rand(3,5,4)).

8. Return the batch matrix-matrix product of a 3D matrix and a 2D matrix (a=torch.rand(3,4,5), b=torch.rand(5,4)).

## Solutions

```
a = torch.rand(3, 3)

a.unsqueeze(0)

a.squeeze(0)

3 + torch.rand(5, 3) * (7 - 3)

a = torch.rand(3, 3)

a.normal_()

a = torch.Tensor([1, 1, 1, 0, 1])

torch.nonzero(a)

a = torch.rand(3, 1)

a.expand(3, 4)

a = torch.rand(3, 4, 5)

b = torch.rand(3, 5, 4)

torch.bmm(a, b)

a = torch.rand(3, 4, 5)

b = torch.rand(5, 4)

torch.bmm(a, b.unsqueeze(0).expand(a.size(0), *b.size()))
```
