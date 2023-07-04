import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 判断是否有gpu可用

path = "./database/ASCAD_fix_key/3-th_key/"
# 导入训练数据，进行预处理
train_traces = np.load(path + "train_traces_45000.npy")
train_labels = np.load(path + "train_labels_45000.npy")
validation_traces = np.load(path + "validation_traces_5000.npy")
validation_labels = np.load(path + "validation_labels_5000.npy")

# 转成张量
train_traces = train_traces.reshape((train_traces.shape[0], 1, train_traces.shape[1]))
train_traces = torch.FloatTensor(train_traces)
train_labels = torch.LongTensor(train_labels)

validation_traces = validation_traces.reshape((validation_traces.shape[0], 1, validation_traces.shape[1]))
validation_traces = torch.FloatTensor(validation_traces)
validation_labels = torch.LongTensor(validation_labels)

batch_size = 200

# 构建训练数据集
train_dataset = TensorDataset(train_traces, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 构建验证数据集
validation_dataset = TensorDataset(validation_traces, validation_labels)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)


class SCA_model(nn.Module):
    """
    功能：根据传入的参数自定义SCA模型
    cnn_layer_num: 卷积层数
    cnn_kernel_size: 卷积核大小
    cnn_stride: 卷积步长
    cnn_padding: 卷积填充
    cnn_pool_size: 池化层大小
    fc_layer_num: 全连接层数
    """

    def __init__(self, cnn_layer_num=1, cnn_kernel_size=11,  cnn_stride=1, cnn_padding=5, cnn_pool_size=2,
                 cnn_activation_function=nn.ReLU(),
                 fc_layer_num=2, fc1_output_number=1024, fc_activation_function=nn.ReLU()):
        super(SCA_model, self).__init__()
        """卷积层"""
        self.conv = nn.Sequential()  # 用于存放卷积层
        self.cnn_layer_num = cnn_layer_num
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_stride = cnn_stride
        self.cnn_padding = cnn_padding
        self.cnn_pool_size = cnn_pool_size
        self.cnn_activation_function = cnn_activation_function
        self.cnn_output_number = 64  # 卷积层输出通道数

        """全连接层"""
        self.fc = nn.Sequential()  # 用于存放全连接层
        self.fc_layer_num = fc_layer_num
        self.fc1_output_number = fc1_output_number
        self.fc_activation_function = fc_activation_function

        """添加卷积层"""
        self.conv.add_module("conv0", nn.Conv1d(1, self.cnn_output_number,
                                                kernel_size=self.cnn_kernel_size,
                                                stride=1,
                                                padding=5))
        self.conv.add_module("conv0_activation_function", self.cnn_activation_function)
        self.conv.add_module("pool0", nn.AvgPool1d(self.cnn_pool_size, self.cnn_pool_size, 0))
        for i in range(1, cnn_layer_num):
            self.conv.add_module("conv" + str(i), nn.Conv1d(self.conv_output_number, self.conv_output_number * 2
                                                            , 11, 1, 5))
            self.conv.add_module("conv" + str(i) + "_activation_function", nn.ReLU())
            self.conv.add_module("pool" + str(i), nn.AvgPool1d(cnn_pool_size, cnn_pool_size, 0))
        self.conv.add_module("flatten", nn.Flatten())

        # 全连接层
        print(self.pool_size, self.cnn_layer_num)
        self.fc.add_module("fc0", nn.Linear(in_features=int(self.conv_output_number * (2 ** (self.cnn_layer_num - 1))
                                                            * (700 / (self.pool_size ** (self.cnn_layer_num + 1)))),
                                            out_features=1024))
        self.fc.add_module("relu" + str(i), nn.ReLU())
        for i in range(1, 3 - 1):  # 减一的目的是让层数名从0开始
            self.fc.add_module("fc" + str(i), nn.Linear(in_features=1024, out_features=512))
            self.fc.add_module("relu" + str(i), nn.ReLU())
        self.fc.add_module("fc" + str(3 - 1), nn.Linear(512, 256))
        self.fc.add_module("logsoftmax", nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


class SCA_model_hyperparameter_search:
    """
    功能：搜索SCA模型的超参数，采用并行搜索
    参数说明：hyperparameters是一个数组，存放多个超参数，每个超参数是一个字典，包含超参数的名称、取值范围、步长
            iterate_number是迭代次数
            initial_points_number是初始候选点的个数
            subspaces_number是划分的超参数搜索子空间的个数，子空间的个数越多，搜索效率越高，但是需要的迭代次数也越多
    """

    def __init__(self, hyperparameters, iterate_number, initial_points_number, subspaces_number):
        self.hyperparameters = hyperparameters  # 存放每个超参数的名称、取值范围、步长、初始候选点
        self.iterate_number = iterate_number
        self.initial_points_number = initial_points_number
        self.subspaces_number = subspaces_number

    def get_initial_points(self):
        """
        功能：根据传入的参数选取初始候选点
        :return: 超参数的初始候选点
        """
        for item in self.hyperparameters:  # 遍历每个超参数 根据迭代周期选取初始候选点
            # 根据超参数的取值范围和步长随机生成初始候选点， 候选点的个数为initial_points_number   \表示换行
            self.hyperparameters[item]['initial_points'] = \
                np.random.randint(
                    low=self.hyperparameters[item]['range'][0] / self.hyperparameters[item]['step'],
                    high=(self.hyperparameters[item]['range'][1] / self.hyperparameters[item]['step']) + 1,
                    size=self.initial_points_number) * self.hyperparameters[item]['step']

        # print(self.hyperparameters)

    def initial_model_training(self):
        """
        功能：训练模型
        :return: 模型参数
        """
        for i in range(0, self.initial_points_number):
            model = SCA_model(cnn_layer_num=self.hyperparameters['cnn_layers_number']['initial_points'][i],
                              cnn_pool_size=self.hyperparameters['cnn_pooling_size']['initial_points'][i])
            model = model.to(device)
            # 打印模型结构
            print(model)

    def construct_parzen_window_PDF(self):
        """
        功能：构建概率密度函数
        :return: 无
        """
        for i in self.hyperparameters:
            # i['parzen_window_PDF'] = self.parzen_window_PDF(i['initial_points'], i['step'])
            pass
        # px = [self.gaussian_kernel(x, mu=mu, sigma=sigma) for mu in data]  # 每个样本点的概率密度  mu为均值 sigma为标准差
        # # print(px)
        # temp = np.mean(np.array(px), axis=0)  # 概率密度的均值  axis=0表示按列求均值
        # # print(temp)
        # return temp

    @staticmethod
    def gaussian_kernel(x, mu, sigma):
        """
        功能：计算高斯分布的概率密度函数
        :参数 mu: 均值
        :参数 sigma: 标准差
        :return:
        """
        probability = 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - mu) ** 2 / 2 / sigma ** 2)
        return probability


# 建议的超参数范围
"""
cnn_layers_number: [1, 5]
cnn_pooling_size: [1, 4]
"""


SCA = SCA_model_hyperparameter_search(
    {
        'cnn_layers_number': {
                'range': [2, 5],
                'step': 1
            },
        'cnn_pooling_size': {
            'range': [1, 4],
            'step': 1
            }
    }, iterate_number=10, initial_points_number=5, subspaces_number=2)

SCA.get_initial_points()
SCA.initial_model_training()
