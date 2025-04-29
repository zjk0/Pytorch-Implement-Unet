import os
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

'''
@brief 训练集的类
'''
class TrainDataset (Dataset):
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
        return output, output_pool  # 前者用于跳跃连接，后者用于下一层的输入

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

'''
@brief 模型训练
@param model: 需要训练的模型
@param dataloader: 数据加载器
@param optimizer: 优化器
@param device: 运行设备
@return loss_list: 一轮epoch的损失函数列表
'''
def train (model, dataloader, optimizer, device):
    model.to(device)
    loss_func = nn.CrossEntropyLoss()  # 损失函数为交叉熵
    loss_list = []

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, targets)
        loss.backward()  # 反向传播
        optimizer.step()

        print(f"当前学习率：{scheduler.get_last_lr()[0]}")
        print(f"损失函数计算值：{loss.item()}")
        loss_list.append(loss.item())

    return loss_list

'''
@brief 模型测试
@param model: 需要测试的模型
@param dataloader: 数据加载器
@param device: 运行设备
@return outputs: 测试结果列表
@return file_names: 测试文件的文件名
'''
def test (model, dataloader, device):
    model.eval()  # 测试模式
    outputs = []  # 输出值列表
    file_names = []  # 用于测试的文件名的列表

    with torch.no_grad():  # 没有梯度计算
        for inputs, file_name in dataloader:
            inputs = inputs.to(device)
            output = model(inputs)
            outputs.append(output)
            file_names.append(file_name)

    return outputs, file_names

if __name__ == '__main__':   
    epoches = 50  # 训练次数
    batch_size = 4  # 批大小
    loss_lists = []  # 保存每轮epoch的loss列表的列表

    # 创建训练集的加载器
    data_train_dir = "./lung_seg_data/img_train"
    label_train_dir = "./lung_seg_data/lab_train"
    lung_data = TrainDataset(data_train_dir, label_train_dir)
    lung_dataloader = DataLoader(lung_data, batch_size = batch_size, shuffle = True)
    
    # 创建测试集的加载器
    data_test_dir = "./lung_seg_data/img_test"
    test_data = TestDataset(data_test_dir)
    test_dataloader = DataLoader(test_data, batch_size = batch_size)

    # 创建一个Unet
    u_net = Unet()

    # Adam优化器
    optimizer = optim.Adam(params = u_net.parameters(), lr = 0.001)

    # 学习率规划器
    scheduler = StepLR(optimizer, 10, 0.5)

    # 模型运行设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 训练
    for epoch in range(epoches):
        loss_list = train(u_net, lung_dataloader, optimizer, device)
        scheduler.step()  # 规划学习率
        loss_lists.append(loss_list)
        torch.cuda.empty_cache()  # 清空缓存
        print(f"完成第{epoch + 1}次训练")

    # 测试
    print("正在测试")
    test_output, test_file_name = test(u_net, test_dataloader, device)
    print("完成测试")

    # 对测试结果进行处理，得到分割结果
    test_output = torch.cat(test_output, dim = 0)
    test_result = torch.argmax(test_output, dim = 1)
    output_array = test_result.cpu().numpy()
    output_array = output_array * 255
    output_array = output_array.astype(np.uint8)

    # 测试结果保存的路径
    test_result_dir = "./test_result"
    os.makedirs(test_result_dir, exist_ok = True)

    # 由于batch_size是4，test函数返回的列表的元素是一个长度为4的元组，现在将元素改为单个文件名
    test_file_name = [name for batch_size in test_file_name for name in batch_size]

    # 写入测试结果
    print("正在写入图片")
    for i in range(output_array.shape[0]):
        path = os.path.join(test_result_dir, test_file_name[i])
        cv.imwrite(path, output_array[i, :, :])
    
    print("写入完成")

    # 保存模型
    save_path = "./unet_model.pth"
    torch.save(u_net.state_dict(), save_path)

    # 绘制每个epoch的loss曲线，画在一张图中
    for i in range(epoches):
        t = range(len(loss_lists[i]))
        plt.plot(t, loss_lists[i])

    plt.show()