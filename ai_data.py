import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost
from astropy.io import fits
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split



class Custom1DDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        Args:
            data (list or numpy array): 一维数据列表或数组
            labels (list or numpy array): 对应的标签列表或数组
            transform (callable, optional): 对数据进行转换的可调用对象
        """
        self.data = torch.tensor(data, dtype=torch.float32)  # 转换为PyTorch张量
        self.labels = torch.tensor(labels, dtype=torch.int)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)  # 如果定义了转换，则应用转换

        label = self.labels[idx]

        return sample, label


class CNNmodel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNNmodel, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.fc1 = nn.Linear(32, 128)
        self.fc1 = nn.Linear(32 * (input_size // 4), 128)  # Assuming pooling reduces size by 4x
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        print('11',x.shape)
        x = self.relu(x)
        x = self.pool(x)
        print('22',x.shape)
        x.flatten()
        # x.view(-1,x.size(2))
        print('33',x.shape)
        # x = x.view(-1, 32 * (x.size(2)))  # Flatten the tensor before the fully connected layer
        x = self.fc1(x)
        print('44',x.shape)
        x = self.relu(x)
        x = self.fc2(x)
        return x



hdulist = fits.open('/data/qcx_data/output/train_data_10.fits')
fluxs = hdulist[0].data
train_data = (fluxs - np.mean(fluxs)) / np.std(fluxs)
# train_data = train_data.reshape(1,train_data.shape[0],train_data.shape[1])
wavelength = np.linspace(3900,9000,3000)
objids = hdulist[1].data['objid']
train_labels = hdulist[1].data['label'].astype(int)

hdulist = fits.open('/data/qcx_data/output/test_data.fits')
fluxs = hdulist[0].data
test_data = (fluxs - np.mean(fluxs)) / np.std(fluxs)
# test_data = test_data.reshape(1,test_data.shape[0],test_data.shape[1])
test_labels = np.zeros(len(test_data))

# train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)
# trainset = Custom1DDataset(train_data, ltrain_labels))
# testset = Custom1DDataset(test_data, list(test_labels))

# X_train_tensor = torch.from_numpy(train_data).float()
# y_train_tensor = torch.from_numpy(train_labels).long()
# X_val_tensor = torch.from_numpy(val_data).float()
# y_val_tensor = torch.from_numpy(val_labels).int()
# X_test_tensor = torch.from_numpy(test_data).float()
# y_test_tensor = torch.from_numpy(test_labels).int()
#
# # 创建数据加载器
# train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
#
#
#
# model = CNNmodel(input_size=3000, num_classes=3)
# model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
#
# # 4. 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
#
# num_epochs = 10  # 假设训练10个epoch
# best_acc = 9.0  # 用于存储最佳验证准确率
#
# for epoch in range(num_epochs):
#     model.train()  # 设置为训练模式
#     running_loss = 0.0
#     corrects = 0
#     total = 0
#
#     for inputs, labels in train_loader:
#         # 清除梯度缓存
#         optimizer.zero_grad()
#
#         # 前向传播
#         outputs = model(inputs.unsqueeze(1))  # 如果输入是一维的，需要增加一个维度作为通道维度
#         loss = criterion(outputs, labels)
#
#         # 反向传播和优化
#         loss.backward()
#         optimizer.step()
#
#         # 统计数据
#         running_loss += loss.item() * inputs.size(0)
#         _, preds = torch.max(outputs, 1)
#         corrects += torch.sum(preds == labels.data)
#         total += labels.size(0)
#
#         # 计算训练集上的平均损失和准确率
#     epoch_loss = running_loss / total
#     epoch_acc = corrects.double() / total
#
#     # 评估模型在验证集上的性能
#     model.eval()  # 设置为评估模式
#     val_loss = 0.0
#     val_corrects = 0
#     val_total = 0
#     with torch.no_grad():  # 不需要计算梯度
#         for inputs, labels in val_loader:
#             inputs = inputs.unsqueeze(1)  # 如果输入是一维的，需要增加一个维度作为通道维度
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             val_loss += loss.item() * inputs.size(0)
#             _, preds = torch.max(outputs, 1)
#             val_corrects += torch.sum(preds == labels.data)
#             val_total += labels.size(0)
#
#             # 计算验证集上的平均损失和准确率
#     val_epoch_loss = val_loss / val_total
#     val_epoch_acc = val_corrects.double() / val_total
#
#     # 打印统计信息
#     print(
#         f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')
#
#     # 保存最佳模型
#     if val_epoch_acc > best_acc:
#         best_acc = val_epoch_acc
#         torch.save(model.state_dict(), 'best_model.pth')
#
#     # 加载最佳模型并测试
# model.load_state_dict(torch.load('best_model.pth'))
# model.eval()
print(train_data.shape,train_labels.shape)
xg_reg = xgboost.XGBClassifier(colsample_bytree = 0.7, learning_rate = 0.01,
                max_depth = 80, alpha = 10, n_estimators = 200)

xg_reg.fit(train_data,train_labels)

y_pre = xg_reg.predict(test_data)

ids= np.linspace(0,len(y_pre)-1,len(y_pre))

result = pd.DataFrame(columns=['objid','label'])
result.objid = ids
result.label = y_pre
result.to_csv('/data/qcx_data/spectra_result.csv',index=False)
