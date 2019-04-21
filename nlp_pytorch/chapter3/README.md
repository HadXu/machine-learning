# chapter3

## The Simplest Neural Network

一个简单的神经网络就是一个线性回归。

```
import torch
import torch.nn as nn

class Perceptron(nn.Module):
    """ A perceptron is one linear layer """
    def __init__(self, input_dim):
        """
        Args:
            input_dim (int): size of the input features
        """
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1)
       
    def forward(self, x_in):
        """The forward pass of the perceptron
        
        Args:
            x_in (torch.Tensor): an input data tensor 
                x_in.shape should be (batch, num_features)
        Returns:
            the resulting tensor. tensor.shape should be (batch,).
        """
        return torch.sigmoid(self.fc1(x_in)).squeeze()
```

## 激活函数

1. sigmoid
2. tanh
3. ReLU
4. PReLU
5. Softmax

## 损失函数

1. Mean Squared Error Loss
```
import torch
import torch.nn as nn

mse_loss = nn.MSELoss()
outputs = torch.randn(3, 5, requires_grad=True)
targets = torch.randn(3, 5)
loss = mse_loss(outputs, targets)
print(loss)
```
2. CrossEntropyLoss
```
import torch
import torch.nn as nn

ce_loss = nn.CrossEntropyLoss()
outputs = torch.randn(3, 5, requires_grad=True)
targets = torch.tensor([1, 0, 3], dtype=torch.int64)
loss = ce_loss(outputs, targets)
print(loss)
```

3. Binary cross-entropy loss
```
bce_loss = nn.BCELoss()
sigmoid = nn.Sigmoid()
probabilities = sigmoid(torch.randn(4, 1, requires_grad=True))
targets = torch.tensor([1, 0, 1, 0],  dtype=torch.float32).view(4, 1)
loss = bce_loss(probabilities, targets)
print(probabilities)
print(loss)
```


## conclusion

1 The weights and bias values are internally managed in the nn.Linear class. If, for some unlikely reason, you would like a model without the bias, you can explicitly set bias=False in the constructor of nn.Linear.

2 There are many types of activation functions—the PyTorch library itself has more than 20 predefined. When you are comfortable with this chapter, you can peruse the documentation to learn more.

3 The words “probability” and “distribution” here must be taken with a grain of salt. By “probability,” what we mean is that the value at outputs is bounded between 0 and 1. By “distribution,” we mean the outputs sum to 1.

4 Two properties are required for a multinomial distribution vector: the sum over elements in the vector should be one and every element in the vector should be nonnegative.

5 In PyTorch, there are actually two softmax functions: Softmax() and LogSoftmax(). LogSoftmax() produces log-probabilities, which preserve the relative ratios of any two numbers but aren’t going to run into numerical problems.

6 This is true only when the base of the log function is the exponential constant e, which is the default base for PyTorch’s log.

7 Using the one-hots in the cross-entropy formula means that all but one of the multiplications will result in a nonzero value. This is a large waste of computation.

8 Note that the code example shows the ground truth vector as being a float vector. Although binary cross entropy is nearly the same as categorical cross entropy (but with only two classes), its computations leverage the 0 and 1 values in the binary cross-entropy formula rather than using them as indexing indices, as was shown for categorical cross entropy.

9 We are sampling from two Gaussian distributions with unit variance. If you don’t get what that means, just assume that the “shape” of the data looks like what’s shown in the figure.

10 There is a perpetual debate in the machine learning and optimization communities on the merits and demerits of SGD. We find that such discussions, although intellectually stimulating, get in the way of learning.

11 You can find the code for classifying the sentiment of Yelp reviews in this book’s GitHub repository.

12 You can find the code for munging the “light” and “full” versions of Yelp review dataset on GitHub.

13 This split of data into training, validation, and test sets works well with large datasets. Sometimes, when the training data is not large, we recommend using k-fold cross validation. How large is “large”? That depends on the network being trained, the complexity of the task being modeled, the size of input instances, and so on, but for many NLP tasks, this is usually when you have hundreds of thousands or millions of training examples.

14 Data cleaning or preprocessing is an important issue that’s glossed over in many machine learning books (and even papers!). We have intentionally kept the concepts simple here to focus more on the modeling, but we highly recommend studying and using all available text preprocessing tools, including NLTK and spaCy. Preprocessing can either improve or hinder accuracy, depending on the data and the task. Use recommendations of what has worked in the past, and experiment often with small data subsets. When implementing a paper, if you find the preprocessing information is missing/unclear, ask the authors!

15 Recall from Chapter 2 that for some languages splitting on whitespace might not be ideal, but we are dealing with cleaned-up English reviews here. You might also want to review “Corpora, Tokens, and Types” at this point.

16 You will see more special tokens when we get to sequence models in Chapter 6

17 Words in any language follow a power law distribution. The number of unique words in the corpus can be on the order of a million, and the majority of these words appear only a few times in the training dataset. Although it is possible to consider them in the model’s vocabulary, doing so will increase the memory requirement by an order of magnitude or more.

18 Recall that in order to subclass PyTorch’s Dataset class, the programmer must implement the __getitem__() and __len__() methods. This allows the DataLoader class to iterate over the dataset by iterating over the indices in the dataset.

19 We use the Namespace class from the built-in argparse package because it nicely encapsulates a property dictionary and works well with static analyzers. Additionally, if you build out command line–based model training routines, you can switch to using the ArgumentParser from the argparse package without changing the rest of your code.
