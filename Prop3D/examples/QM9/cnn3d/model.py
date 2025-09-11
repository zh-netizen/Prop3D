
import torch
import torch.nn as nn

class InceptionDWConv3d(nn.Module):
    """
    Inception-style 3D Depthwise Convolution Module.

    Splits the input channels into five parts: one identity branch and four depthwise convolution branches
    with different kernel orientations to capture spatial features along different dimensions.

    Args:
        in_channels (int): Number of input channels.
        cube_kernel_size (int): Kernel size for the cubic depthwise convolution.
        band_kernel_size (int): Kernel size for band-shaped convolutions along specific dimensions.
        branch_ratio (float): Ratio to split input channels for each convolution branch.

    Forward Input:
        x (Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

    Forward Output:
        Tensor of the same shape as input, with features concatenated from identity and four depthwise conv branches.
    """
    def __init__(self, in_channels, cube_kernel_size=3, band_kernel_size=5, branch_ratio=0.125):
        super().__init__()
        gc = int(in_channels * branch_ratio)
        # Depthwise conv on cubic kernel (D, H, W)
        self.dwconv_hwd = nn.Conv3d(gc, gc, cube_kernel_size, padding=cube_kernel_size // 2, groups=gc)
        # Depthwise conv along W dimension
        self.dwconv_wd = nn.Conv3d(gc, gc, kernel_size=(1, 1, band_kernel_size), padding=(0, 0, band_kernel_size // 2), groups=gc)
        # Depthwise conv along H dimension
        self.dwconv_hd = nn.Conv3d(gc, gc, kernel_size=(1, band_kernel_size, 1), padding=(0, band_kernel_size // 2, 0), groups=gc)
        # Depthwise conv along D dimension
        self.dwconv_hw = nn.Conv3d(gc, gc, kernel_size=(band_kernel_size, 1, 1), padding=(band_kernel_size // 2, 0, 0), groups=gc)
        # Define how to split channels for each branch
        self.split_indexes = (in_channels - 4 * gc, gc, gc, gc, gc)

    def forward(self, x):
        # Split input tensor into identity and four convolution branches
        x_id, x_hwd, x_wd, x_hd, x_hw = torch.split(x, self.split_indexes, dim=1)
        # Concatenate identity and processed branches along channel dimension
        x = torch.cat(
            (x_id, self.dwconv_hwd(x_hwd), self.dwconv_wd(x_wd), self.dwconv_hd(x_hd), self.dwconv_hw(x_hw)),
            dim=1
        )
        return x


class CBAM3d(nn.Module):
    """
    3D Convolutional Block Attention Module (CBAM).

    Applies sequential channel and spatial attention mechanisms to enhance feature representation.

    Args:
        in_channels (int): Number of input channels.
        reduction (int): Reduction ratio for the channel attention module.

    Forward Input:
        x (Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

    Forward Output:
        Tensor of same shape as input, with applied attention weights.
    """
    def __init__(self, in_channels, reduction=16):
        super(CBAM3d, self).__init__()

        # Channel Attention Module: adaptive average pooling -> MLP -> sigmoid weights
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # Output shape: (B, C, 1, 1, 1)
            nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Spatial Attention Module: apply max and average pooling across channel dimension, then conv + sigmoid
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Apply channel attention
        x = x * self.channel_attention(x)

        # Apply spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Channel-wise average pooling
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Channel-wise max pooling
        spatial_input = torch.cat([avg_out, max_out], dim=1)  # Concatenate for spatial attention
        x = x * self.spatial_attention(spatial_input)

        return x


class RegressionModel(nn.Module):
    """
    3D Convolutional Regression Model with Inception-style Depthwise Convolutions and CBAM attention modules.

    Architecture:
    - Initial 3D convolution to expand channels.
    - Three InceptionDWConv3d blocks, each followed by a CBAM module.
    - Two max-pooling layers for downsampling.
    - Fully connected layers to output regression prediction.

    Args:
        in_channels (int): Number of input channels.
        out_features (int): Number of output features (default is 1 for regression).

    Forward Input:
        x (Tensor): Input tensor of shape (batch_size, in_channels, D, H, W), where D=H=W=16.

    Forward Output:
        Tensor of shape (batch_size,) containing predicted regression values.
    """
    def __init__(self, in_channels=5, out_features=1):
        super().__init__()
        # Initial convolution to expand input channels to 64
        self.expand_conv = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
        self.cbam_expand = CBAM3d(64)  # CBAM attention after expansion conv
        self.bn1 = nn.BatchNorm3d(64)

        # First inception block + attention
        self.inception_conv1 = InceptionDWConv3d(64, 3, 5, 0.125)
        self.cbam_inception1 = CBAM3d(64)

        # Downsample
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Second inception block + attention
        self.inception_conv2 = InceptionDWConv3d(64, 3, 5, 0.125)
        self.cbam_inception2 = CBAM3d(64)

        # Third inception block + attention
        self.inception_conv3 = InceptionDWConv3d(64, 3, 5, 0.125)
        self.cbam_inception3 = CBAM3d(64)

        # Downsample again
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Flatten and fully connected layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 4 * 4 * 4, 512)
        self.fc1 = nn.Linear(512, out_features)

    def forward(self, x):
        x = self.expand_conv(x)
        x = self.cbam_expand(x)  # Apply CBAM
        x = self.bn1(x)

        x = self.inception_conv1(x)
        x = self.cbam_inception1(x)  # Apply CBAM
        x = self.pool1(x)

        x = self.inception_conv2(x)
        x = self.cbam_inception2(x)  # Apply CBAM

        x = self.inception_conv3(x)
        x = self.cbam_inception3(x)  # Apply CBAM
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc(x)
        x = self.fc1(x)

        # Return output reshaped as 1D tensor for regression
        return x.view(-1)

