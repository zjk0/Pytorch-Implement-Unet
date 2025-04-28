import argparse
import os

import cv2 as cv
import numpy as np

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from PIL import ImageTk

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

'''
@brief 训练集的类
'''
class LungDataset (Dataset):
    def __init__ (self, data_train_dir, label_train_dir):
        super().__init__()

        self.data_train_dir = data_train_dir
        self.label_train_dir = label_train_dir

        self.data_train_list = os.listdir(data_train_dir)
        self.label_train_list = os.listdir(label_train_dir)

    def __len__ (self):
        # 返回训练集长度
        return len(self.data_train_list)
    
    def __getitem__ (self, index):
        # 得到图像路径
        data_train_path = os.path.join(self.data_train_dir, self.data_train_list[index])
        label_train_path = os.path.join(self.label_train_dir, self.label_train_list[index])

        # 读取图像，得到numpy数组
        data_train = cv.imread(data_train_path, cv.IMREAD_GRAYSCALE)
        label_train = cv.imread(label_train_path, cv.IMREAD_GRAYSCALE)

        # 统一大小
        data_train = cv.resize(data_train, (512, 512))
        label_train = cv.resize(label_train, (512, 512))

        # 归一化
        data_train = data_train / 255.0
        label_train = label_train / 255.0

        # 转换为pytorch的张量形式
        data_train_tensor = torch.from_numpy(data_train).float().unsqueeze(0)
        label_train_tensor = torch.from_numpy(label_train).long()

        return data_train_tensor, label_train_tensor

'''
@brief 测试集的类
'''
class TestDataset (Dataset):
    def __init__ (self, data_test_dir):
        super().__init__()

        self.data_test_dir = data_test_dir
        self.data_test_list = os.listdir(data_test_dir)

    def __len__ (self):
        # 返回测试集长度
        return len(self.data_test_list)
    
    def __getitem__ (self, index):
        data_test_path = os.path.join(self.data_test_dir, self.data_test_list[index])  # 得到图像路径
        data_test = cv.imread(data_test_path, cv.IMREAD_GRAYSCALE)  # 读取图像，得到numpy数组
        data_test = data_test / 255.0  # 归一化
        data_test_tensor = torch.from_numpy(data_test).float().unsqueeze(0)  # 转换为pytorch的张量形式

        return data_test_tensor, self.data_test_list[index]

'''
@brief Unet编码器的一层的类
'''
class UnetEncoderBlock (nn.Module):
    def __init__ (self, in_channels, out_channels, conv_size = 3, conv_stride = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels, 
                out_channels = out_channels, 
                kernel_size = conv_size, 
                stride = conv_stride, 
                padding = int((conv_size - 1) / 2)
            ),
            nn.ReLU(), 
            nn.Conv2d(
                in_channels = out_channels, 
                out_channels = out_channels, 
                kernel_size = conv_size, 
                stride = conv_stride, 
                padding = int((conv_size - 1) / 2)
            ),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(kernel_size = 2)

    def forward (self, input):
        output = self.block(input)
        output_pool = self.pool(output)
        return output, output_pool

'''
@brief Unet解码器的一层的类
'''
class UnetDecoderBlock (nn.Module):
    def __init__ (self, in_channels, out_channels, conv_size = 3, conv_stride = 1):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(
            in_channels = in_channels, 
            out_channels = out_channels, 
            kernel_size = 2, 
            stride = 2
        )

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels, 
                out_channels = out_channels, 
                kernel_size = conv_size, 
                stride = conv_stride, 
                padding = int((conv_size - 1) / 2)
            ),
            nn.ReLU(), 
            nn.Conv2d(
                in_channels = out_channels, 
                out_channels = out_channels, 
                kernel_size = conv_size, 
                stride = conv_stride, 
                padding = int((conv_size - 1) / 2)
            ),
            nn.ReLU()
        )

    def forward (self, input, encoder_output):
        input_up_conv = self.up_conv(input)
        connect = torch.cat([input_up_conv, encoder_output], dim = 1)
        output = self.block(connect)
        return output

'''
@brief Unet类
'''
class Unet (nn.Module):
    def __init__ (self):
        super().__init__()

        self.encoder1 = UnetEncoderBlock(1, 64)
        self.encoder2 = UnetEncoderBlock(64, 128)
        self.encoder3 = UnetEncoderBlock(128, 256)
        self.encoder4 = UnetEncoderBlock(256, 512)

        self.mid_trans = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size = 3, padding = 1),
            nn.ReLU()
        )

        self.decoder1 = UnetDecoderBlock(1024, 512)
        self.decoder2 = UnetDecoderBlock(512, 256)
        self.decoder3 = UnetDecoderBlock(256, 128)
        self.decoder4 = UnetDecoderBlock(128, 64)

        self.final_trans = nn.Conv2d(64, 2, kernel_size = 1)

    def forward (self, input):
        en_output_1, en_output_1_pool = self.encoder1(input)
        en_output_2, en_output_2_pool = self.encoder2(en_output_1_pool)
        en_output_3, en_output_3_pool = self.encoder3(en_output_2_pool)
        en_output_4, en_output_4_pool = self.encoder4(en_output_3_pool)

        mid_trans_output = self.mid_trans(en_output_4_pool)

        de_output_1 = self.decoder1(mid_trans_output, en_output_4)
        de_output_2 = self.decoder2(de_output_1, en_output_3)
        de_output_3 = self.decoder3(de_output_2, en_output_2)
        de_output_4 = self.decoder4(de_output_3, en_output_1)

        final_trans_output = self.final_trans(de_output_4)

        return final_trans_output

def train (module, dataloader, device):
    module.to(device)
    optimizer = Adam(params = module.parameters(), lr = 0.001)
    loss_func = nn.CrossEntropyLoss()

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = module(inputs)
        loss = loss_func(outputs, targets)
        loss.backward()
        optimizer.step()

        print(f"损失函数计算值：{loss.item()}")

def test (module, dataloader, device):
    module.eval()
    outputs = []
    file_names = []

    with torch.no_grad():
        for inputs, file_name in dataloader:
            inputs = inputs.to(device)
            output = module(inputs)
            outputs.append(output)
            file_names.append(file_name)

    return outputs, file_names

if __name__ == '__main__':   
    epoches = 5  # 训练次数
    batch_size = 4  # 批大小

    # 创建训练集的加载器
    data_train_dir = "./lung_seg_data/img_train"
    label_train_dir = "./lung_seg_data/lab_train"
    lung_data = LungDataset(data_train_dir, label_train_dir)
    lung_dataloader = DataLoader(lung_data, batch_size = batch_size, shuffle = True)
    
    # 创建测试集的加载器
    data_test_dir = "./lung_seg_data/img_test"
    test_data = TestDataset(data_test_dir)
    test_dataloader = DataLoader(test_data, batch_size = batch_size)

    # 创建一个Unet
    u_net = Unet()

    # 模型运行设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 训练
    for epoch in range(epoches):
        train(u_net, lung_dataloader, device)
        torch.cuda.empty_cache()  # 清空缓存
        print(f"完成第{epoch + 1}次训练")

    print("正在测试")
    test_output, test_file_name = test(u_net, test_dataloader, device)
    print("完成测试")

    test_output = torch.cat(test_output, dim = 0)
    test_result = torch.argmax(test_output, dim = 1)
    output_array = test_result.cpu().numpy()
    output_array = output_array * 255
    output_array = output_array.astype(np.uint8)

    test_result_dir = "./test_result"
    os.makedirs(test_result_dir, exist_ok = True)

    test_file_name = [name for batch_size in test_file_name for name in batch_size]
    print("正在写入图片")
    for i in range(output_array.shape[0]):
        path = os.path.join(test_result_dir, test_file_name[i])
        cv.imwrite(path, output_array[i, :, :])
    
    print("写入完成")