import torch.nn as nn


class MLP(nn.Module):
    """
    Parameters:
    -----------
    input_size: int
        Number of input features
    hidden_sizes: list
        List of hidden layer sizes
    output_size: int
        Number of output features
    activation: str
        Activation function to use between layers
    """

    def __init__(self, input_size, hidden_sizes, num_classes, activation='relu'):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.activation = activation
        self.layers = nn.ModuleList()
        self.build()

    def build(self):
        sizes = [self.input_size] + self.hidden_sizes + [self.num_classes]
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                if self.activation == 'relu':
                    self.layers.append(nn.ReLU())
                elif self.activation == 'tanh':
                    self.layers.append(nn.Tanh())
                elif self.activation == 'sigmoid':
                    self.layers.append(nn.Sigmoid())
                else:
                    raise NotImplementedError

    def forward(self, x):
        
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        return f"MLP(input_size={self.input_size}, hidden_sizes={self.hidden_sizes}, output_size={self.output_size}, activation={self.activation})"

    def __str__(self):
        return self.__repr__()
