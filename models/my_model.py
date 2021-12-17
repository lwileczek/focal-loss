import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        #############################################################################
        #       Initialize the network weights                                      #
        #############################################################################
        # Ideas & help came from: 
        #   - https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
        #   - https://arxiv.org/pdf/1412.6071.pdf
        #   - https://www.cs.toronto.edu/~kriz/conv-cifar10-aug2010.pdf
        # Convolutional Layers, Add second layer to find even more abstract features
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=4)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2)
        # ReLU (Leaky didn't work so well), research paper said to limit relu since there
        # are only a few classes and not to over fit. 
        self.activation = nn.ReLU6()
        # another paper on this data set found improvments with this pooling strategy
        self.pool = nn.FractionalMaxPool2d(3, output_ratio=(0.7, 0.7))
        self.drop = nn.Dropout(p=0.2)  # Regularization
        self.fc = nn.Linear(51200, 10)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        #       Implement forward pass of the network                               #
        #############################################################################
        x = self.drop(self.pool(self.activation(self.conv0(x))))
        x = self.drop(self.activation(self.conv1(x)))
        x = self.drop(self.pool(self.activation(self.conv2(x))))
        # Re-org from Tensor to matrix (n-dimentional to 2-dimentional)
        s = x.size()
        hidden_size = 1
        for d in s[1:]:
            hidden_size *= d
        # print("\nHidden Size:", hidden_size, "\n\n")
        x = x.view(-1, hidden_size)
        # Fully connected layer 
        outs  = self.fc(x)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs
