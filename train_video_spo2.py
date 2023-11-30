import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
import cv2
import pandas as pd
import numpy as np

from physnet_train.PhysNetModel import PhysNet
from resnet_train.new.dataset import scaler_input, inverse_transform, scaler_label
from resnet_train.new.resnet import ResNet, ResidualBlock
from physnet_train.utils_sig import hr_fft, butter_bandpass

# 1. 数据准备

video_folder = '20230830视频血氧/crop'  # 视频文件夹路径
label_folder = '20230830视频血氧/Data'  # 标签文件路径
device = torch.device('cuda')



# 定义存储视频帧和标签的列表
frames = []
labels = []
print('----------------------')
# 遍历视频文件夹中的所有视频文件
# 遍历读取标签数据
for video_file in os.listdir(video_folder):
    video_path = os.path.join(video_folder, video_file)
    temp = video_file.split('.')[0] + '.csv'
    label_file = os.path.join(label_folder, temp)
    labels_df = pd.read_csv(label_file)
    spo2_labels = labels_df.iloc[:, 2]
    spo2_labels = spo2_labels.tolist()
    labels.append(spo2_labels)
    # 读取视频文件并提取帧图像
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = cv2.resize(frame, (128, 128))
        frames.append(frame)
    cap.release()

    # # 获取标签值
    # video_name = os.path.splitext(video_file)[0]
    # label = labels_df.loc[labels_df['video_name'] == video_name, 'label'].values[0]
    # labels.append(label)
print('000000000000000000')
#将大列表分割成小的批次再进行张量转化
batch_size = 10
input_tensor_list = []
target_tensor_list = []
# for i in range(0, len(frames), batch_size):
#     batch_data = frames[i:i+batch_size]
#     temp = torch.tensor(batch_data, dtype=torch.float32) / 255.0
#     print('////////////////////'+str(i))
#     input_tensor_list.append(temp)
# 转换为张量并归一化
# input_tensor = torch.cat(input_tensor_list, dim=0)
# torch.save(input_tensor, 'input.pt')
print('123')
labels = [item for sublist in labels for item in sublist]
target_tensor = torch.tensor(labels, dtype=torch.float32)
print('456')

torch.save(target_tensor, 'target.pt')
# 数据配对
input_tensor = torch.load('input.pt')
# 数据对齐
input_tensor = input_tensor[:31290,:,:,:]
input_tensor = input_tensor.reshape(1043,30,128,128,3)
#维度位置变化呢
input_tensor = input_tensor.permute(0,4,1,2,3)
out_len = target_tensor.size(0)
indices_to_reduce = torch.randperm(out_len)[:10]
#创建布尔类型掩码
mask = torch.ones(out_len, dtype=torch.bool)
mask[indices_to_reduce] = False
target_tensor = target_tensor[mask]
assert input_tensor.shape[0] == target_tensor.shape[0], "Input and target sizes mismatch"
dataset = TensorDataset(input_tensor, target_tensor)

# 划分训练集和验证集
train_ratio = 1
train_size = int(train_ratio * len(dataset))
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
# 定义大模型
class BigModel(nn.Module):
    def __init__(self):
        super(BigModel, self).__init__()
        self.model1 = PhysNet().to(device)
        self.model2 = ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=1).to(device)
        # 添加更多的子模型...

    def forward(self, x, fps):
        #增加一个维度ｂａｔｃｈ
        x = x.to(device)
        out1 = self.model1(x)
        out1 = out1.to(device)
        out1er = out1[:, -1, :]
        rppg = out1er[0].cpu().detach().numpy()
        rppg = butter_bandpass(rppg, lowcut=0.6, highcut=4, fs=fps)
        temp = out1[:, 0:3, :]
        #模型２输入（Ｂ，Ｔ，Ｃ）
        temp = temp.permute(0, 2, 1)
        temp = temp.cpu().detach().numpy()
        two_scale = np.reshape(temp,(temp.shape[0]*temp.shape[1],temp.shape[2]))
        two_scale = scaler_input.fit_transform(two_scale)
        three_scale = np.reshape(two_scale, (temp.shape[0], temp.shape[1], temp.shape[2]))
        temp = torch.tensor(three_scale).to(device)
        out2_list = []

        # 进一步处理子模型的输出...
        for i in range(int(temp.size()[1]/30)):
            sliced_tensor = temp[:,30*i:30*i+30, :]
            out2 = self.model2(sliced_tensor)
            out2_list.append(out2.detach().cpu().numpy())
        out2_list = np.vstack(out2_list)

        out2_list = inverse_transform(out2_list, scaler_label)

        return rppg, out2_list

model = BigModel()
model.model1.load_state_dict(torch.load(r'/home/lbx/下载/contrast-phys-master/results/5/epoch29.pt'))
model.model2.load_state_dict(torch.load(r'/home/lbx/下载/contrast-phys-master/all/resnet_train/new/lstm_model.pth'))

# 3. 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.00001)


# 4. 训练过程
device = torch.device("cuda")
model.to(device)

# 数据加载器
batch_size = 7
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)

# 训练循环
epochs = 1000
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    j = 0
    for inputs, targets in train_loader:
        j+=1
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # 前向传播
        rppg, pred_freq_spo2 = model(inputs, 30.0)


        pred_freq_spo2 = torch.tensor(pred_freq_spo2)
        pred_freq_spo2 = pred_freq_spo2.to('cuda')
        print("+++++++++++++++++++"+str(j))
        print(pred_freq_spo2)
        print(targets)
        loss = criterion(pred_freq_spo2, targets)
        print(loss)
        # 反向传播和优化
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)


    train_loss /= len(train_set)

    # 在每个训练周期结束后，使用验证集评估模型性能
    # model.eval()
    # val_loss = 0.0
    # with torch.no_grad():
    #     for inputs, targets in val_loader:
    #         inputs = inputs.to(device)
    #         targets = targets.to(device)
    #
    #         outputs = model(inputs)
    #         rppg = outputs[:, -1, :]
    #         rppg = rppg[0].detach().cpu().numpy()
    #         rppg = butter_bandpass(rppg, lowcut=0.6, highcut=4, fs=30)
    #         pred_freq, psd_y, psd_x = hr_fft(rppg, fs=30)  # 自定义函数，用于计算频率
    #         loss = criterion(pred_freq, targets)
    #
    #         val_loss += loss.item() * inputs.size(0)
    #
    # val_loss /= len(val_set)

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}")
    if epoch%10==0:
        torch.save(model.state_dict(), 'hr_model_%d.pt'%epoch)
# 5. 模型应用
# TODO: 使用训练好的模型进行预测和应用