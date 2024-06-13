import random
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import csv

import paddle
import paddle.vision.transforms as T
from paddle.io import Dataset, DataLoader

# 按比例随机切割数据集 训练集占0.9，验证集占0.1
train_ratio = 0.9  

train_paths, train_labels = [], []
valid_paths, valid_labels = [], []
with open('data/train_list.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if random.uniform(0, 1) < train_ratio:
            train_paths.append(line.split('\t')[0])
            label = line.split('\t')[1]
            train_labels.append(int(label))
        else:
            valid_paths.append(line.split('\t')[0])
            valid_labels.append(int(label))

# 定义训练数据集
class TrainData(Dataset):
    def __init__(self):
        super().__init__()
        self.color_jitter = T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05)
        self.normalize = T.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
        self.random_crop = T.RandomCrop(224, pad_if_needed=True)
    
    def __getitem__(self, index):
        # 读取图片
        image_path = train_paths[index]
        image = np.array(Image.open(image_path))  
        if image.ndim == 2: 
            image = np.stack([image] * 3, axis=-1)
        image = image.transpose([2, 0, 1])  
        
        # 图像增广
        features = self.color_jitter(image.transpose([1, 2, 0]))
        features = self.random_crop(features)
        features = self.normalize(features.transpose([2, 0, 1])).astype(np.float32)

        # 读取标签
        labels = train_labels[index]

        return features, labels
    
    def __len__(self):
        return len(train_paths)

# 定义验证数据集
class ValidData(Dataset):
    def __init__(self):
        super().__init__()
        self.normalize = T.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
    
    def __getitem__(self, index):
        # 读取图片
        image_path = valid_paths[index]
        image = np.array(Image.open(image_path)) 
        if image.ndim == 2:  
            image = np.stack([image] * 3, axis=-1)
        image = image.transpose([2, 0, 1])  
        
        # 图像变换
        features = cv2.resize(image.transpose([1, 2, 0]), (256, 256)).transpose([2, 0, 1]).astype(np.float32)
        features = self.normalize(features)

        # 读取标签
        labels = valid_labels[index]

        return features, labels
    
    def __len__(self):
        return len(valid_paths)

# 调用resnet50模型
paddle.vision.set_image_backend('cv2')
model = paddle.vision.models.resnet50(pretrained=True, num_classes=12)

# 定义数据迭代器
train_data = TrainData()
valid_data = ValidData()
train_dataloader = DataLoader(train_data, batch_size=256, shuffle=True, drop_last=False)

# 定义优化器
opt = paddle.optimizer.Adam(learning_rate=1e-4, parameters=model.parameters(), weight_decay=paddle.regularizer.L2Decay(1e-4))

# 定义损失函数
loss_fn = paddle.nn.CrossEntropyLoss()
paddle.set_device('gpu:0')

# 整体训练流程
for epoch_id in range(15):
    model.train()
    for batch_id, data in enumerate(train_dataloader()):
    
        features, labels = data
        features = paddle.to_tensor(features)
        labels = paddle.to_tensor(labels)
        predicts = model(features)
        loss = loss_fn(predicts, labels)
        avg_loss = paddle.mean(loss)
        avg_loss.backward()

        opt.step()
        opt.clear_grad()
        if batch_id % 2 == 0:
            print('epoch_id:{}, batch_id:{}, loss:{}'.format(epoch_id, batch_id, avg_loss.numpy()))
    
    model.eval()
    print('开始评估')
    i = 0
    acc = 0
    for image, label in valid_data:
        image = paddle.to_tensor([image])
        pre = list(np.array(model(image)[0]))
        max_item = max(pre)
        pre = pre.index(max_item)

        i += 1
        if pre == label:
            acc += 1
        if i % 10 == 0:
            print('精度：', acc / i)
    
    paddle.save(model.state_dict(), 'acc{}.model'.format(acc / i))

# 进行预测和提交
def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
test_path = []
listdir('data/cat_12_test', test_path)

# 加载训练好的模型
pre_model = paddle.vision.models.resnet50(pretrained=True, num_classes=12)
pre_model.set_state_dict(paddle.load('acc0.9285714285714286.model'))
pre_model.eval()

pre_classes = []
normalize = T.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
# 生成预测结果
for path in test_path:
    image_path = path

    image = np.array(Image.open(image_path))  
    if image.ndim == 2:  
        image = np.stack([image] * 3, axis=-1)
    image = image.transpose([2, 0, 1])  
    
    features = cv2.resize(image.transpose([1, 2, 0]), (256, 256)).transpose([2, 0, 1]).astype(np.float32)
    features = normalize(features)

    features = paddle.to_tensor([features])
    pre = list(np.array(pre_model(features)[0]))
    max_item = max(pre)
    pre = pre.index(max_item)
    print("图片：", path, "预测结果：", pre)
    pre_classes.append(pre)

print(pre_classes)

# 创建提交文件
with open('submit.csv', 'w', encoding='gbk', newline="") as f:
    csv_writer = csv.writer(f)
    for i in range(len(test_path)):
        csv_writer.writerow([os.path.basename(test_path[i]), pre_classes[i]])
    print('写入数据完成')
