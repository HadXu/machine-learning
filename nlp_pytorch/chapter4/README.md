# chapter4: Feed-Forward Networks for Natural Language Processing

第三章我们考虑的为一层简单神经网络，深度学习就是多层神经网络的统称。

```
import torch.nn as nn
import torch.nn.functional as F

class MultilayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Args:
            input_dim (int): the size of the input vectors
            hidden_dim (int): the output size of the first Linear layer
            output_dim (int): the output size of the second Linear layer
        """
        super(MultilayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_in, apply_softmax=False):
        """The forward pass of the MLP
        
        Args:
            x_in (torch.Tensor): an input data tensor 
                x_in.shape should be (batch, input_dim)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the cross-entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, output_dim)
        """
        intermediate = F.relu(self.fc1(x_in))
        output = self.fc2(intermediate)
        
        if apply_softmax:
            output = F.softmax(output, dim=1).
        return output
```


## CNN

```
class SurnameClassifier(nn.Module):
    def __init__(self, initial_num_channels, num_classes, num_channels):
        """
        Args:
            initial_num_channels (int): size of the incoming feature vector
            num_classes (int): size of the output prediction vector
            num_channels (int): constant channel size to use throughout network
        """
        super(SurnameClassifier, self).__init__()
        
        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=initial_num_channels, 
                      out_channels=num_channels, kernel_size=3),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, 
                      kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, 
                      kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, 
                      kernel_size=3),
            nn.ELU()
        )
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x_surname, apply_softmax=False):
        """The forward pass of the classifier
        
        Args:
            x_surname (torch.Tensor): an input data tensor
                x_surname.shape should be (batch, initial_num_channels,
                                           max_surname_length)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the cross-entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, num_classes).
        """
        features = self.convnet(x_surname).squeeze(dim=2)
        prediction_vector = self.fc(features)

        if apply_softmax:
            prediction_vector = F.softmax(prediction_vector, dim=1)

        return prediction_vector
```


## Network-in-Network Connections (1x1 Convolutions)

## Residual Connections/Residual Block

# Summary

In this chapter, you learned two basic feed-forward architectures: the multilayer perceptron (MLP; also called “fully-connected” network) and the convolutional neural network (CNN). We saw the power of MLPs in approximating any nonlinear function and showed applications of MLPs in NLP with the application of classifying nationalities from surnames. We studied one of the major disadvantages/limitations of MLPs—lack of parameter sharing—and introduced the convolutional network architecture as a possible solution. CNNs, originally developed for computer vision, have become a mainstay in NLP; primarily because of their highly efficient implementation and low memory requirements. We studied different variants of convolutions—padded, dilated, and strided—and how they transform the input space. This chapter also dedicated a nontrivial length of discussion on the practical matter of choosing input and output sizes for the convolutional filters. We showed how the convolution operation helps capture substructure information in language by extending the surname classification example to use convnets. Finally, we discussed some miscellaneous, but important, topics related to convolutional network design: 1) Pooling, 2) BatchNorm, 3) 1x1 convolutions, and 4) residual connections. In modern CNN design, it is common to see many of these tricks employed at once as seen in the Inception architecture (Szegedy et al., 2015) in which a mindful use of these tricks led convolutional networks hundreds of layers deep that were not only accurate but fast to train. In the Chapter 5, we explore the topic of learning and using representations for discrete units, like words, sentences, documents, and other feature types using Embeddings.

