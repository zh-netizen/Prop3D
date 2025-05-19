import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


class CNN3D_SMP(nn.Module):
    def __init__(self, in_channels, spatial_size,
                 conv_drop_rate, fc_drop_rate,
                 conv_filters, conv_kernel_size,
                 max_pool_positions, max_pool_sizes, max_pool_strides,
                 fc_units,
                 batch_norm=True,
                 dropout=False):
        super(CNN3D_SMP, self).__init__()

        layers = []
        if batch_norm:
            layers.append(nn.BatchNorm3d(in_channels))

        # Convs
        for i in range(len(conv_filters)):#添加4个3dcnn
            layers.extend([
                nn.Conv3d(in_channels, conv_filters[i],
                          kernel_size=conv_kernel_size,
                          bias=True),
                nn.ReLU()
                ])
            spatial_size -= (conv_kernel_size - 1)
            if max_pool_positions[i]:#池化
                layers.append(nn.MaxPool3d(max_pool_sizes[i], max_pool_strides[i]))
                spatial_size = int(np.floor((spatial_size - (max_pool_sizes[i]-1) - 1)/max_pool_strides[i] + 1))
            if batch_norm:#归一化
                layers.append(nn.BatchNorm3d(conv_filters[i]))
            if dropout:#dropout
                layers.append(nn.Dropout(conv_drop_rate))
            in_channels = conv_filters[i]

        layers.append(nn.Flatten())
        in_features = in_channels * (spatial_size**3)
        # FC layers
        for units in fc_units:
            layers.extend([
                nn.Linear(in_features, units),
                nn.ReLU()
                ])
            if batch_norm:
                layers.append(nn.BatchNorm3d(units))
            if dropout:
                layers.append(nn.Dropout(fc_drop_rate))
            in_features = units

        # Final FC layer
        layers.append(nn.Linear(in_features, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).view(-1)


print('*********************juanji********************************************')


class InceptionDWConv3d(nn.Module):
    def __init__(self, in_channels, cube_kernel_size=3, band_kernel_size=5, branch_ratio=0.125):
        super().__init__()
        gc = int(in_channels * branch_ratio)
        self.dwconv_hwd = nn.Conv3d(gc, gc, cube_kernel_size, padding=cube_kernel_size // 2, groups=gc)
        self.dwconv_wd = nn.Conv3d(gc, gc, kernel_size=(1, 1, band_kernel_size), padding=(0, 0, band_kernel_size // 2),
                                   groups=gc)
        self.dwconv_hd = nn.Conv3d(gc, gc, kernel_size=(1, band_kernel_size, 1), padding=(0, band_kernel_size // 2, 0),
                                   groups=gc)
        self.dwconv_hw = nn.Conv3d(gc, gc, kernel_size=(band_kernel_size, 1, 1), padding=(band_kernel_size // 2, 0, 0),
                                   groups=gc)
        self.split_indexes = (in_channels - 4 * gc, gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hwd, x_wd, x_hd, x_hw = torch.split(x, self.split_indexes, dim=1)
        x = torch.cat(
            (x_id, self.dwconv_hwd(x_hwd), self.dwconv_wd(x_wd), self.dwconv_hd(x_hd), self.dwconv_hw(x_hw)),
            dim=1
        )
        return x




# class CBAM3d(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(CBAM3d, self).__init__()
#         self.channel_attention = nn.Sequential(
#             # nn.AvgPool3d(kernel_size=(1, 1, 1), stride=1, padding=0),
#             nn.
#             nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0),
#             nn.Sigmoid()
#         )
#         # 修改空间注意力机制的卷积核大小为3x3x3
#         self.spatial_attention = nn.Sequential(
#             nn.Conv3d(in_channels, 1, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         x = x * self.channel_attention(x)
#         x = x * self.spatial_attention(x)
#         return x

class CBAM3d(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM3d, self).__init__()

        # 通道注意力机制：用自适应平均池化
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # 对空间维度做自适应池化，输出形状 (B, C, 1, 1, 1)
            nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()  # 输出权重用于乘法
        )

        # 空间注意力机制：先对通道做平均池化和最大池化，再拼接输入
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False),  # 输入2通道（平均池化+最大池化）
            nn.Sigmoid()  # 输出空间注意力权重
        )

    def forward(self, x):
        # 通道注意力机制
        x = x * self.channel_attention(x)

        # 空间注意力机制
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 平均池化，得到 (B, 1, D, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 最大池化，得到 (B, 1, D, H, W)
        spatial_input = torch.cat([avg_out, max_out], dim=1)  # 拼接 (B, 2, D, H, W)
        x = x * self.spatial_attention(spatial_input)  # 空间注意力加权

        return x


class RegressionModel(nn.Module):
    def __init__(self, in_channels=5, out_features=1):
        super().__init__()
        # 输入: (batch_size, in_channels, D, H, W)  D=H=W=16
        self.expand_conv = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
        self.cbam_expand = CBAM3d(64)  # 为 expand_conv 添加 CBAM
        self.bn1 = nn.BatchNorm3d(64)

        self.inception_conv1 = InceptionDWConv3d(64, 3, 5, 0.125)
        self.cbam_inception1 = CBAM3d(64)  # 为 inception_conv1 添加 CBAM

        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.inception_conv2 = InceptionDWConv3d(64, 3, 5, 0.125)
        self.cbam_inception2 = CBAM3d(64)  # 为 inception_conv2 添加 CBAM

        self.inception_conv3 = InceptionDWConv3d(64, 3, 7, 0.125)
        self.cbam_inception3 = CBAM3d(64)  # 为 inception_conv3 添加 CBAM

        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        self.fc = nn.Linear(64 * 4 * 4 * 4, 512)
        self.fc1 = nn.Linear(512, out_features)

    def forward(self, x):
        x = self.expand_conv(x)
        x = self.cbam_expand(x)  # 应用 CBAM 模块
        x = self.bn1(x)
        x = self.inception_conv1(x)
        x = self.cbam_inception1(x)  # 应用 CBAM 模块
        x = self.pool1(x)
        x = self.inception_conv2(x)
        x = self.cbam_inception2(x)  # 应用 CBAM 模块
        x = self.inception_conv3(x)
        x = self.cbam_inception3(x)  # 应用 CBAM 模块
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.fc1(x)
        return x.view(-1)