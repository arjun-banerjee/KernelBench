import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Dilated convolution tower with exponentially increasing dilation rates and residual connections.
    Each block applies LayerNorm -> GELU -> DilatedConv1d, then adds back the original input as a residual.
    This architecture is a core building block of genomic foundation models (Enformer, Basenji, Caduceus)
    for capturing multi-scale sequence dependencies in DNA/RNA.
    """
    def __init__(self, channels, seq_len, kernel_size, max_dilation_power):
        super(Model, self).__init__()
        self.norms = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.activation = nn.GELU()
        for i in range(max_dilation_power + 1):
            dilation = 2 ** i
            self.norms.append(nn.LayerNorm([channels, seq_len]))
            self.convs.append(nn.Conv1d(channels, channels, kernel_size, padding=dilation, dilation=dilation))

    def forward(self, x):
        residual = x
        for norm, conv in zip(self.norms, self.convs):
            x = norm(x)
            x = self.activation(x)
            x = conv(x)
            x = x + residual
        return x

batch_size = 32
channels = 256
seq_len = 4096
kernel_size = 3
max_dilation_power = 8

def get_inputs():
    return [torch.randn(batch_size, channels, seq_len)]

def get_init_inputs():
    return [channels, seq_len, kernel_size, max_dilation_power]