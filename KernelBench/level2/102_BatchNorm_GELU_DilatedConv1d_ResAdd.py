import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Pre-norm 1D dilated convolution block with BatchNorm: BatchNorm1d, ReLU activation, dilated Conv1d, and residual addition.
    At high dilation rates, memory access becomes non-contiguous across large strides in the sequence dimension.
    """
    def __init__(self, channels, kernel_size, dilation):
        super(Model, self).__init__()
        self.norm = nn.BatchNorm1d(channels)
        self.activation = nn.ReLU()
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=dilation, dilation=dilation)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)
        x = x + residual
        return x

batch_size = 32
channels = 256
seq_len = 4096
kernel_size = 3
dilation = 128

def get_inputs():
    return [torch.randn(batch_size, channels, seq_len)]

def get_init_inputs():
    return [channels, kernel_size, dilation]