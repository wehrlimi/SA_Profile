import torch
import torch.nn as nn

from models import *


class MaxBlurPool(nn.Module):
    """
    Shift-invariant alternative to MaxPool (https://arxiv.org/abs/1904.11486)
    """
    def __init__(self, channels, stride=2):
        super(MaxBlurPool, self).__init__()
        self.stride = stride
        
        # Binomial-5 filter: [1, 4, 6, 4, 1]
        kernel = torch.tensor([1., 4., 6., 4., 1.])
        kernel = kernel[:, None] * kernel[None, :] # outer product for 2D kernel
        kernel = kernel / kernel.sum() # normalize
        self.register_buffer('kernel', kernel[None, None, :, :].repeat(channels, 1, 1, 1))
    
    def forward(self, x):
        x = F.max_pool2d(x, kernel_size=self.stride, stride=1, padding=0)
        x = F.conv2d(x, self.kernel, stride=self.stride, padding=2, groups=x.shape[1])  # Padding=2 for 5x5 kernel
        return x


class ConvBlock(nn.Module):
    """
    A convolutional block with optional residual connection
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        layers (int): Number of mid-layer convolutional layers
        residual (bool): Whether to use a residual connection
        dropout_p (float, optional): Dropout probability for regularization (default: 0.0).
    """
    def __init__(self, in_channels, out_channels, layers, residual, dropout_p=0.):
        super(ConvBlock, self).__init__()
        if layers < 2:
            raise ValueError(f"Argument 'layers' has to be at least 2.")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual = residual
        dilations = [1] * layers if isinstance(layers, int) else layers
        
        mid_channels = max(in_channels, out_channels)
        channels = [in_channels] + [mid_channels] * (layers - 1) + [out_channels]
        
        if residual:
            self.conv_skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
        self.convs = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(channels[i], channels[i+1], kernel_size=3, dilation=dilations[i], padding=dilations[i], bias=False, padding_mode='replicate'),
                nn.InstanceNorm2d(channels[i+1]),
                nn.ReLU()
            ) for i in range(layers - 1)
        ] + [
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(channels[-2], channels[-1], kernel_size=3, dilation=dilations[-1], padding=dilations[-1], bias=True, padding_mode='replicate')
        ])
    
    def forward(self, x):
        if self.residual:
            return self.conv_skip(x) + self.convs(x)
        else:
            return self.convs(x)
    

class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        #self.pool = nn.MaxPool2d(kernel_size=factor, stride=factor)
        self.pool = MaxBlurPool(channels=channels, stride=2)
    
    def forward(self, x):
        return self.pool(x)


class UpSample(nn.Module):
    def __init__(self):
        super(UpSample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        return self.upsample(x)



class UNet(nn.Module):
    """
    U-Net with many customization options

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        down_channels (list[int]): Number of channels for each downsampling block.
        down_layers (list[int] | list[list[int]]): Number of 3x3 convolutional layers for each downsampling block.
        up_channels (list[int]): Number of channels for each upsampling block.
        up_layers (list[int] | list[list[int]]): Number of 3x3 convolutional layers for each upsampling block.
        bottleneck_channels (int): Number of channels in the bottleneck block.
        bottleneck_layers (list[int]): Number of 3x3 convolutional layer in the bottleneck block.
        keypoint_extraction (str, optional): Method to extract coordinates from predicted heatmaps; 'argmax' or 'weighted' (default: 'argmax').
        residual (bool, optional): Whether to use a residual connection in the convolution blocks (default: True).
        dropout_p (float, optional): Dropout probability for regularization (default: 0.0).
    """
    def __init__(self, in_channels, out_channels, down_channels, down_layers, up_channels, up_layers, bottleneck_channels, bottleneck_layers,
                 keypoint_extraction="argmax", residual=True, dropout_p=0.):
        super(UNet, self).__init__()
        self.init_args = {k: v for k, v in locals().items() if k not in ["self", "__class__"]}

        self.out_channels = out_channels
        self.keypoint_extraction = keypoint_extraction

        self.num_levels = len(down_channels)

        self.downconv = nn.ModuleList([
            ConvBlock(in_channels if i==0 else down_channels[i-1], down_channels[i], layers=down_layers[i], residual=residual, dropout_p=dropout_p)
            for i in range(self.num_levels)
        ])
        self.downsample = nn.ModuleList([ DownSample(down_channels[i]) for i in range(self.num_levels) ])

        self.bottleneck = ConvBlock(down_channels[-1], bottleneck_channels, layers=bottleneck_layers, residual=residual, dropout_p=dropout_p)

        self.upsample = nn.ModuleList([ UpSample() for i in range(self.num_levels-1, -1, -1) ])
        self.upconv = nn.ModuleList([
            ConvBlock(in_channels=(bottleneck_channels if i==0 else up_channels[i-1]) + down_channels[-(i+1)],
                  out_channels=up_channels[i], layers=up_layers[i], residual=residual, dropout_p=dropout_p)
            for i in range(self.num_levels)
        ])

        self.conv_out = nn.Conv2d(in_channels=up_channels[-1], out_channels=out_channels, kernel_size=1, padding=0, bias=False)

        self.apply(kaiming_weight_init)

    def forward(self, x):
        enc = []

        for i in range(self.num_levels):
            x = self.downconv[i](x)
            enc.append(x)
            x = self.downsample[i](x)

        x = self.bottleneck(x)

        for i in range(self.num_levels):
            x = self.upsample[i](x)
            skip = enc.pop()
            x = self.upconv[i](torch.cat([x, skip], dim=1))
        
        out = self.conv_out(x)

        return out

    def predict(self, x):
        with torch.no_grad():
            out = softmax2d(self.forward(x)).detach().cpu()

        return { "keypoints" : extract_keypoints(out, method=self.keypoint_extraction), "heatmaps" : out }


