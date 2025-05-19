from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import numpy as np
import pytest
from torch.utils.data import Dataset, DataLoader
import scipy
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
def read_griddata(pname):
    x = []
    y = []
    smiles = []
    shape_set = set()

    descriptor_data, grid_shape = joblib.load(pname)
    shape_set.add(grid_shape)
    if len(shape_set) != 1:  # the shape in each file must be equal
        raise ValueError('Grid shapes of the folds are not compatible.')
    for desc in descriptor_data:
        x.append(desc[0])
        y.append(desc[1])
        smiles.append(desc[2])
    # return np.array(x), np.array(y), np.array(smiles), shape_set.pop()
    return x,y,smiles,shape_set.pop()

grid_x, grid_y, grid_smiles, sample_shape = read_griddata('/home/zhanghuan/atom3d/examples/esol/grid3Dmols_tox21_SR-ATAD5_rotate')
print(len(grid_x))


def create_grid_from_coordinates(grid_x, grid_shape):
    """
    根据给定的坐标列表 (grid_x)，在 3D 网格中对应位置的通道上设置为 1。

    grid_x: list of tuples, 每个元组包含 (x, y, z, channel)
    grid_shape: 网格的形状, tuple (depth, height, width, channels)

    返回填充好的 3D 网格。
    """
    # 初始化一个零值的网格，形状为 (depth, height, width, channels)
    grid = np.zeros(grid_shape, dtype=int)

    # 遍历坐标列表，设置对应位置的通道值为 1
    for coord in grid_x:
        x, y, z, channel = coord

        # 检查坐标是否在网格的范围内
        if 0 <= x < grid_shape[0] and 0 <= y < grid_shape[1] and 0 <= z < grid_shape[2] and 0 <= channel < grid_shape[
            3]:
            # 设置对应位置的通道为 1
            grid[x, y, z, channel] = 1

    return grid

'''
你可以通过字典的键来访问数据，例如：
data_dict[0]['grid'] 获取第一个样本的网格数据
data_dict[0]['label'] 获取第一个样本的标签
data_dict[0]['smile'] 获取第一个样本的 SMILES 字符串
'''

def get_grid(grid_x,grid_y,grid_shape,grid_smiles):
    all_dic={}
    zuobiao_list=grid_x
    labe_list=grid_y
    smile_list=grid_smiles
    grid_shape=grid_shape
    for i in range(len(grid_x)):
        print(f'当前为第{i}个元素')
        grid=create_grid_from_coordinates(zuobiao_list[i],grid_shape)
        label=labe_list[i]
        smile=smile_list[i]
        all_dic[i]={
            'grid':grid,
            'label':label,
            'smile':smile
        }
    return all_dic
import numpy as np

def shuffle_and_split_dict(big_dict, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, random_seed=None):
    """
    打乱大字典中的数据，并按指定比例划分为训练集、验证集和测试集。

    参数:
    big_dict (dict): 包含多个小字典的大字典。每个小字典包含样本的数据。
    train_ratio (float): 训练集所占的比例，默认为 0.8。
    valid_ratio (float): 验证集所占的比例，默认为 0.1。
    test_ratio (float): 测试集所占的比例，默认为 0.1。
    random_seed (int, optional): 用于打乱数据的随机数种子。默认为 None，即不设置种子。

    返回:
    tuple: 返回三个字典，分别是训练集、验证集和测试集。
    """

    # 验证比例和数据总长度是否合理
    total_ratio = train_ratio + valid_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError("train_ratio + valid_ratio + test_ratio must sum to 1.")

    # 设置随机种子，以便复现打乱方式
    if random_seed is not None:
        np.random.seed(random_seed)

    # 先获取大字典中的所有小字典
    items = list(big_dict.values())

    # 打乱所有小字典
    np.random.shuffle(items)

    # 根据给定的比例划分数据集
    total_len = len(items)
    train_size = int(total_len * train_ratio)
    valid_size = int(total_len * valid_ratio)

    train_data = items[:train_size]
    valid_data = items[train_size:train_size + valid_size]
    test_data = items[train_size + valid_size:]

    # 重新构造划分后的字典
    def construct_dict(data):
        return {i: sample for i, sample in enumerate(data)}

    # 返回三个字典
    train_dict = construct_dict(train_data)
    valid_dict = construct_dict(valid_data)
    test_dict = construct_dict(test_data)

    return train_dict, valid_dict, test_dict



class myDataset(Dataset):
    def __init__(self, data_dict):
        """
        :param data_dict: 你的数据字典，其中包含网格、标签、SMILES 等数据。
        例如：{0: {'grid': ..., 'label': ..., 'smile': ...}, ...}
        """
        self.data_dict = data_dict

    def __len__(self):
        """
        返回数据集的大小
        """
        return len(self.data_dict)

    def __getitem__(self, idx):
        """
        获取指定索引的数据样本。
        :param idx: 数据集中的索引
        :return: 一个包含网格数据、标签和 SMILES 字符串的字典
        """
        sample = self.data_dict[idx]

        grid = sample['grid']  # 获取网格
        label = sample['label']  # 获取标签
        smile = sample['smile']  # 获取 SMILES 字符串

        # 将网格转换为 Tensor（如果需要）
        grid = torch.tensor(grid, dtype=torch.float32)  # 假设你需要 float32 类型
        label = torch.tensor(label, dtype=torch.float32)  # 标签类型根据任务可能是 float32 或 int64

        return {'grid': grid, 'label': label, 'smile': smile}

all_dic=get_grid(grid_x,grid_y,(48,48,48,9),grid_smiles)
train_data,valid_data,test_data=shuffle_and_split_dict(all_dic,0.8,0.1,0.1,42)

train_set=myDataset(train_data)
val_set=myDataset(valid_data)
test_set=myDataset(test_data)

train_loader = DataLoader(train_set, batch_size=10, shuffle=True)
val_loader = DataLoader(val_set, batch_size=10, shuffle=False)
test_loader = DataLoader(test_set, batch_size=10, shuffle=False)

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

'''普通3dcnn'''
# class My3DConvNet(nn.Module):
#     def __init__(self):
#         super(My3DConvNet, self).__init__()
#
#         # 定义3D卷积层和池化层
#         self.conv1 = nn.Conv3d(in_channels=9, out_channels=32, kernel_size=3, padding=1)  # 输入通道数9，输出通道数32
#         self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  # 池化层，降采样
#
#         self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # 第二层卷积
#         self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)  # 池化层
#
#         self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1)  # 第三层卷积
#         self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)  # 池化层
#
#         # 全连接层，用于回归任务输出
#         self.fc1 = nn.Linear(128 * 6 * 6 * 6, 512)  # 计算经过卷积层和池化层后特征图的维度
#         self.fc2 = nn.Linear(512, 1)  # 输出一个标量，回归任务的结果
#
#         # 激活函数
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = x.permute(0, 4, 1, 2, 3)  # 将输入从 [batch_size, 48, 48, 48, 9] 转为 [batch_size, 9, 48, 48, 48]
#         # 卷积层+池化层+ReLU激活
#         x = self.relu(self.conv1(x))
#         x = self.pool1(x)
#
#         x = self.relu(self.conv2(x))
#         x = self.pool2(x)
#
#         x = self.relu(self.conv3(x))
#         x = self.pool3(x)
#
#         # 展平特征图，使用 reshape 而不是 view
#         x = x.reshape(x.size(0), -1)  # 扁平化为(batch_size, features)
#
#         # 全连接层
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)  # 输出回归值
#
#         return x.view(-1)

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

'''不使用注意力的3dcnn'''
# class My3DConvNet(nn.Module):#     def __init__(self):
#         super(My3DConvNet, self).__init__()
#
#         # 使用 InceptionDWConv3d 替代标准3D卷积
#         self.conv1 = nn.Conv3d(in_channels=9, out_channels=128, kernel_size=3, padding=1)  # 输入通道数9，输出通道数
#         self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  # 池化层，降采样
#
#         self.inception2 = InceptionDWConv3d(in_channels=128)  # 输入通道数为32（Inception1的输出通道数）
#         self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)  # 池化层
#
#         self.inception3 = InceptionDWConv3d(in_channels=128)  # 输入通道数为128（Inception2的输出通道数）
#         self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)  # 池化层
#
#         # 全连接层，调整输入大小
#         self.fc1 = nn.Linear(128 * 6 * 6 * 6, 512)  # 假设经过卷积和池化后的特征图大小为 (6, 6, 6)
#         self.fc2 = nn.Linear(512, 1)  # 输出一个标量，用于回归任务
#
#         # ReLU激活函数
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         # 输入数据的形状是 [batch_size, 48, 48, 48, 9]，需要调整维度顺序
#         x = x.permute(0, 4, 1, 2, 3)  # 从 [batch_size, 48, 48, 48, 9] 调整为 [batch_size, 9, 48, 48, 48]
#
#         # 第1个 InceptionDWConv3d + 池化 + 激活
#         x = self.relu(self.conv1(x))
#         x = self.pool1(x)
#
#         # 第2个 InceptionDWConv3d + 池化 + 激活
#         x = self.relu(self.inception2(x))
#         x = self.pool2(x)
#
#         # 第3个 InceptionDWConv3d + 池化 + 激活
#         x = self.relu(self.inception3(x))
#         x = self.pool3(x)
#
#         # 扁平化特征图
#         x = x.reshape(x.size(0), -1)  # 展平为 (batch_size, 特征数)
#
#         # 全连接层
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)  # 输出回归值
#
#         return x.view(-1)  # 返回一个一维的输出，用于回归任务

'''使用注意力的3dcnn'''
# class My3DConvNet(nn.Module):
#     def __init__(self):
#         super(My3DConvNet, self).__init__()
#
#         # 定义第1个卷积层
#         self.conv1 = nn.Conv3d(in_channels=9, out_channels=128, kernel_size=3, padding=1)
#         self.cbam1 = CBAM3d(in_channels=128)  # 添加CBAM3d模块
#         self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
#
#         # 定义第2个卷积层（使用InceptionDWConv3d替代传统卷积）
#         self.inception2 = InceptionDWConv3d(in_channels=128)
#         self.cbam2 = CBAM3d(in_channels=128)  # 添加CBAM3d模块
#         self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
#
#         # 定义第3个卷积层（使用InceptionDWConv3d）
#         self.inception3 = InceptionDWConv3d(in_channels=128)
#         self.cbam3 = CBAM3d(in_channels=128)  # 添加CBAM3d模块
#         self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
#
#         # 全连接层
#         self.fc1 = nn.Linear(128 * 6 * 6 * 6, 512)  # 需要根据输入大小调整
#         self.fc2 = nn.Linear(512, 1)  # 输出一个回归值
#
#         # 激活函数
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         # 调整输入数据形状 [batch_size, 48, 48, 48, 9] -> [batch_size, 9, 48, 48, 48]
#         x = x.permute(0, 4, 1, 2, 3)
#
#         # 第1层：卷积 -> CBAM3d -> 池化 -> ReLU
#         x = self.relu(self.conv1(x))
#         x = self.cbam1(x)  # 应用CBAM3d模块
#         x = self.pool1(x)
#
#         # 第2层：InceptionDWConv3d -> CBAM3d -> 池化 -> ReLU
#         x = self.relu(self.inception2(x))
#         x = self.cbam2(x)  # 应用CBAM3d模块
#         x = self.pool2(x)
#
#         # 第3层：InceptionDWConv3d -> CBAM3d -> 池化 -> ReLU
#         x = self.relu(self.inception3(x))
#         x = self.cbam3(x)  # 应用CBAM3d模块
#         x = self.pool3(x)
#
#         # 展平特征图
#         x = x.reshape(x.size(0), -1)  # 展平为 (batch_size, 特征数)
#
#         # 全连接层
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)  # 输出回归值
#
#         return x.view(-1)  # 返回一个标量的输出，用于回归任务
import torch
import torch.nn as nn

class My3DConvNet(nn.Module):
    def __init__(self):
        super(My3DConvNet, self).__init__()

        # 定义第1个卷积层
        self.conv1 = nn.Conv3d(in_channels=9, out_channels=128, kernel_size=3, padding=1)
        self.cbam1 = CBAM3d(in_channels=128)  # 添加CBAM3d模块
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        # 定义第2个卷积层（使用InceptionDWConv3d替代传统卷积）
        self.inception2 = InceptionDWConv3d(in_channels=128)
        self.cbam2 = CBAM3d(in_channels=128)  # 添加CBAM3d模块
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        # 定义第3个卷积层（使用InceptionDWConv3d）
        self.inception3 = InceptionDWConv3d(in_channels=128)
        self.cbam3 = CBAM3d(in_channels=128)  # 添加CBAM3d模块
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Dropout 层
        self.dropout = nn.Dropout(p=0.25)  # p = 0.5, 即有50%的概率丢弃神经元

        # 全连接层
        self.fc1 = nn.Linear(128 * 6 * 6 * 6, 512)  # 需要根据输入大小调整
        self.fc2 = nn.Linear(512, 1)  # 输出一个标量值，用于二分类任务

        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数用于输出

    def forward(self, x):
        # 调整输入数据形状 [batch_size, 48, 48, 48, 9] -> [batch_size, 9, 48, 48, 48]
        x = x.permute(0, 4, 1, 2, 3)

        # 第1层：卷积 -> CBAM3d -> 池化 -> ReLU
        x = self.relu(self.conv1(x))
        x = self.cbam1(x)  # 应用CBAM3d模块
        x = self.pool1(x)

        # 第2层：InceptionDWConv3d -> CBAM3d -> 池化 -> ReLU
        x = self.relu(self.inception2(x))
        x = self.cbam2(x)  # 应用CBAM3d模块
        x = self.pool2(x)

        # 第3层：InceptionDWConv3d -> CBAM3d -> 池化 -> ReLU
        x = self.relu(self.inception3(x))
        x = self.cbam3(x)  # 应用CBAM3d模块
        x = self.pool3(x)

        # 展平特征图
        x = x.reshape(x.size(0), -1)  # 展平为 (batch_size, 特征数)

        # Dropout 层：防止过拟合
        x = self.dropout(x)

        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # 输出一个标量值，用于二分类任务

        # Sigmoid 激活函数输出概率值
        x = self.sigmoid(x)

        return x.view(-1)


from sklearn.metrics import roc_auc_score, average_precision_score
def train(model, train_loader, criterion, optimizer, device):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    all_labels = []
    all_outputs = []

    for data in tqdm(train_loader, desc="Training"):
        # 获取数据
        inputs = data['grid'].to(device)
        labels = data['label'].to(device)

        # 零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)

        # 计算损失 (已使用 Sigmoid 输出)
        loss = criterion(outputs.squeeze(), labels.float())  # 输出为 (batch_size, 1)，标签为 (batch_size, )

        # 反向传播
        loss.backward()

        # 优化
        optimizer.step()

        # 累积损失
        running_loss += loss.item()

        # 保存预测结果和真实标签，用于计算评估指标
        all_labels.append(labels.cpu().numpy())
        all_outputs.append(outputs.cpu().detach().numpy())  # 保存概率值（已经经过 Sigmoid 激活）

    # 合并所有批次的标签和输出
    all_labels = np.concatenate(all_labels, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)

    # 计算 AUROC 和 AUPRC
    auroc = roc_auc_score(all_labels, all_outputs)
    auprc = average_precision_score(all_labels, all_outputs)

    # 计算并返回平均损失和评估指标
    avg_loss = running_loss / len(train_loader)
    return avg_loss, auroc, auprc


def validate(model, val_loader, criterion, device):
    model.eval()  # 设置模型为评估模式
    running_loss = 0.0
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for data in tqdm(val_loader, desc="Validating"):
            # 获取数据
            inputs = data['grid'].to(device)
            labels = data['label'].to(device)

            # 前向传播
            outputs = model(inputs)

            # 计算损失 (已使用 Sigmoid 输出)
            loss = criterion(outputs.squeeze(), labels.float())  # 输出为 (batch_size, 1)，标签为 (batch_size, )

            # 累积损失
            running_loss += loss.item()

            # 保存标签和预测输出
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().detach().numpy())  # 保存概率值（已经经过 Sigmoid 激活）

    # 合并所有批次的标签和输出
    all_labels = np.concatenate(all_labels, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)

    # 计算 AUROC 和 AUPRC
    auroc = roc_auc_score(all_labels, all_outputs)
    auprc = average_precision_score(all_labels, all_outputs)

    # 计算并返回平均损失、AUROC 和 AUPRC
    avg_loss = running_loss / len(val_loader)
    return avg_loss, auroc, auprc


def mytest(model, test_loader, criterion, device):
    model.eval()  # 设置模型为评估模式
    running_loss = 0.0
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing"):
            # 获取数据
            inputs = data['grid'].to(device)
            labels = data['label'].to(device)

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs.squeeze(), labels.float())  # 输出为 (batch_size, 1)，标签为 (batch_size, )

            # 累积损失
            running_loss += loss.item()

            # 保存标签和预测输出
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().detach().numpy())

    # 合并所有批次的标签和输出
    all_labels = np.concatenate(all_labels, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)

    # 计算 AUROC 和 AUPRC
    auroc = roc_auc_score(all_labels, all_outputs)
    auprc = average_precision_score(all_labels, all_outputs)

    # 计算并返回平均损失、AUROC 和 AUPRC
    avg_loss = running_loss / len(test_loader)
    return avg_loss, auroc, auprc


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'训练使用的')

# 初始化模型
model = My3DConvNet().to(device)

# 定义损失函数（使用 BCELoss）和优化器（Adam）
criterion = torch.nn.BCELoss()  # 使用 BCELoss 作为损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用 Adam 优化器

num_epochs = 40  # 设置训练的轮数
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # 训练模型
    train_loss, train_auroc, train_auprc = train(model, train_loader, criterion, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}, Train AUROC: {train_auroc:.4f}, Train AUPRC: {train_auprc:.4f}")

    # 验证模型
    val_loss, val_auroc, val_auprc = validate(model, val_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}, Validation AUROC: {val_auroc:.4f}, Validation AUPRC: {val_auprc:.4f}")

    # 测试模型
    test_loss, test_auroc, test_auprc = mytest(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test AUROC: {test_auroc:.4f}, Test AUPRC: {test_auprc:.4f}")

