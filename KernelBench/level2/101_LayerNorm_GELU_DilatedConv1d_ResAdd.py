import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Pre-norm 1D dilated convolution block: LayerNorm, GELU activation, dilated Conv1d, and residual addition.
    Common in genomic sequence models (Enformer, Basenji) for capturing long-range dependencies in DNA/RNA sequences.
    """
    def __init__(self, channels, seq_len, kernel_size, dilation):
        super(Model, self).__init__()
        self.norm = nn.LayerNorm([channels, seq_len])
        self.activation = nn.GELU()
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
dilation = 4

def get_inputs():
    return [torch.randn(batch_size, channels, seq_len)]

def get_init_inputs():
    return [channels, seq_len, kernel_size, dilation]