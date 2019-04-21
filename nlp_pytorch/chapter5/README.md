# chapter5

When implementing natural language processing tasks, we need to deal with different kinds of discrete types. The most obvious example is words. Words come from a finite set (aka vocabulary). Other examples of discrete types include characters, part-of-speech tags, named entities, named entity types, parse features, items in a product catalog, and so on. Essentially, when any input feature comes from a finite (or a countably infinite) set, it is a discrete type.

> 使用Embedding,可以将高维的特征映射为低维特征(数值特征),使用该特征进行下一步的任务。

## Word Embedding方法

1. Given a sequence of words, predict the next word. This is also called the language modeling task.
2. Given a sequence of words before and after, predict the missing word.
3. Given a word, predict words that occur within a window, independent of the position.

```
import numpy as np
from annoy import AnnoyIndex

class PreTrainedEmbeddings(object):
    def __init__(self, word_to_index, word_vectors):
        """
        Args:
            word_to_index (dict): mapping from word to integers
            word_vectors (list of numpy arrays)
        """
        self.word_to_index = word_to_index
        self.word_vectors = word_vectors
        self.index_to_word = \
            {v: k for k, v in self.word_to_index.items()}
        self.index = AnnoyIndex(len(word_vectors[0]),
                                metric='euclidean')
        for _, i in self.word_to_index.items():
            self.index.add_item(i, self.word_vectors[i])
        self.index.build(50)
    
    @classmethod
    def from_embeddings_file(cls, embedding_file):
        """Instantiate from pretrained vector file.
        
        Vector file should be of the format:
            word0 x0_0 x0_1 x0_2 x0_3 ... x0_N
            word1 x1_0 x1_1 x1_2 x1_3 ... x1_N
        
        Args:
            embedding_file (str): location of the file
        Returns:
            instance of PretrainedEmbeddings
        """
        word_to_index = {}
        word_vectors = []
        with open(embedding_file) as fp:
            for line in fp.readlines():
                line = line.split(" ")
                word = line[0]
                vec = np.array([float(x) for x in line[1:]])
                
                word_to_index[word] = len(word_to_index)
                word_vectors.append(vec)
        return cls(word_to_index, word_vectors)
        
embeddings = PreTrainedEmbeddings.from_embeddings_file('glove.6B.100d.txt')
```


## CBOW

> The CBOW model is a multiclass classification task represented by scanning over texts of words, creating a context window of words, removing the center word from the context window, and classifying the context window to the missing word. Intuitively, you can think of it like a fill-in-the-blank task. There is a sentence with a missing word, and the model’s job is to figure out what that word should be.

给定周围6个单词，预测中间的词

X | Y
:-: | :-:
or the modern by mary wollstonecraft | prometheus
by mary wollstonecraft shelley letter st | godwin

```
class CBOWClassifier(nn.Module): # Simplified cbow Model
    def __init__(self, vocabulary_size, embedding_size, padding_idx=0):
        """
        Args:
            vocabulary_size (int): number of vocabulary items, controls the
                number of embeddings and prediction vector size
            embedding_size (int): size of the embeddings
            padding_idx (int): default 0; Embedding will not use this index
        """
        super(CBOWClassifier, self).__init__()
        
        self.embedding =  nn.Embedding(num_embeddings=vocabulary_size, 
                                       embedding_dim=embedding_size,
                                       padding_idx=padding_idx)
        self.fc1 = nn.Linear(in_features=embedding_size,
                             out_features=vocabulary_size)

    def forward(self, x_in, apply_softmax=False):
        """The forward pass of the classifier
        
        Args:
            x_in (torch.Tensor): an input data tensor. 
                x_in.shape should be (batch, input_dim)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, output_dim)
        """
        x_embedded_sum = F.dropout(self.embedding(x_in).sum(dim=1), 0.3)
        y_out = self.fc1(x_embedded_sum)
        
        if apply_softmax:
            y_out = F.softmax(y_out, dim=1)
            
        return y_out
```

训练采用一般的
```loss_func = nn.CrossEntropyLoss()```即可。

## 输入输出

输入的为[bs, input_dim], 输出的为[bs, vocabulary_size], 表示的为每一个单词的概率，其中
```
F.dropout(self.embedding(x_in).sum(dim=1)
```
原本输出为[bs, input_dim, vocabulary_size],但是我们需要的是整个input_dim个单词加起来的概率。
因此需要进行```.sum(dim=1)```。


