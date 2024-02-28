import cv2  # open cv
from tqdm import tqdm  # 进度条库
import glob  # 返回文件路径中所有匹配的文件名
import os
from PIL import Image, ImageStat  # 图像处理的库
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.resnet import resnet18
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import warnings
import torch.nn.functional as nf
import time
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import d2lzh_pytorch as d2l
# from lib2to3.pgen2.grammar import opmap_raw
# from msilib.schema import Patch
import torch
from torch import nn, optim, tensor
# import sys
import csv
# sys.path.append("..")

warnings.filterwarnings(action='ignore')


# 调用pytorch中的resnet18网络进行预训练权重

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16
lr, num_epoch = 0.001, 100

# 数据预处理部分--以文件夹PMR_0112中的数据为例
model_folder = 'model1'
Path_All = './{}/'.format(model_folder)
train_path = './{}/train/'.format(model_folder)
test_path = './{}/test/'.format(model_folder)
train_txt_path = './{}/train.txt'.format(model_folder)
test_txt_path = './{}/test.txt'.format(model_folder)
Output_path = Path_All + 'Output'.format(model_folder)
log_dir = model_folder + "/Output/log"

# 生成训练和测试使用的标签文件

train_all = {}
test_all = {}
for trainable in [True,False]:
    if trainable:
        for category in ['pic1','pic2','pic3','pic4','pic5']:
            train_all[category] = glob.glob(os.path.join(train_path + category,'*.jpg'))   #这种写法是分了两个文件夹
            print(os.path.join(train_path + category,'*.jpg'))
    else:
        for category in ['pic1','pic2','pic3','pic4','pic5']:
            test_all[category] = glob.glob(os.path.join(test_path + category,'*.jpg'))
#结果显示有几个图片尺寸与其他的不符,手动删除它
###制作数据集对应的标签
mapkey = {
    'pic1' : '0',
    'pic2' : '1',
    'pic3' : '2',
    'pic4' : '3',
    'pic5' : '4',
}

def gen_txt(txt_path, img_paths):
    f = open(txt_path,'w')
    for key in img_paths.keys():
        label = mapkey[key]
        for path in img_paths[key]:
            line = path + ' ' + label +'\n'
            f.write(line)

gen_txt(train_txt_path,train_all)
gen_txt(test_txt_path,test_all)

## 生成输出结果的文件夹
if not os.path.exists(Output_path):  # 如果val下没有子文件夹，就创建
    os.makedirs(Output_path)

# 构建Dataset子类
class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()  # 删除字符串末尾的指定字符，默认为空格
            words = line.split()  # 分割字符串，默认以空格为分隔符进行全分割
            img_path = words[0]
            img_path = img_path.replace('\\','/') # linux中路径“\\”无法工作
            img_label = int(words[1])
            # print(words[0],'\n',words[1],type(words))
            imgs.append((img_path, img_label))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        # convert将图像转化为对应的像素值，有三种模式1、RGB，三通道。2、1,01模式，3、L模式，将每个像素点转化为0-255，单通道
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)

train_paths = train_all['pic1'] + train_all['pic2'] + train_all['pic3'] + train_all['pic4']+ train_all['pic5']
m_list, s_list = [], []
print('Read pic: \n')
for path in tqdm(train_paths):  # tqdm进度条模式，挺有意思的
    img = cv2.imread(path)
    img = img / 255.0
    m, s = cv2.meanStdDev(img)
    m_list.append(m.reshape((3,)))
    s_list.append(s.reshape((3,)))
m_array = np.array(m_list)
s_array = np.array(s_list)
# 转换为矩阵
m = m_array.mean(axis=0, keepdims=True)
s = s_array.mean(axis=0, keepdims=True)

# print(m[0][::-1])
# print(s[0][::-1])
# 数据预处理，编写transform 进行数据增强
normMean = m[0][::-1].copy()
normStd = s[0][::-1].copy()
# 索引中出现了负数的话，最好加一个copy()，虽然我也还没弄明白为啥

#normMean = [0.48827705, 0.45510637, 0.41741   ]
#normStd = [0.22971935, 0.22475049, 0.22525084]
normTransform = transforms.Normalize(normMean, normStd)
trainTransform = transforms.Compose([
    #  transforms.RandomRotation(30),
    # 随机旋转 角度为30度
    # transforms.RandomHorizontalFlip(),
    # 图片旋转,水平或者垂直方向
    # transforms.Resize((280,280)),
    # transforms.CenterCrop((1900,990)),
    # transforms.Resize((40, 30)),
    # 按比例将图像进行缩放至指定尺寸
    transforms.ToTensor(),
    # 转换为Tensor
    normTransform,
    # 再进行归一化
])

testTransform = transforms.Compose([
    # transforms.Resize((280,280)),
    # transforms.CenterCrop((1900,990)),
    # transforms.Resize((40, 30)),
    transforms.ToTensor(),
    normTransform,
])
# 制作train_loader test_loader
train_data = MyDataset(txt_path=train_txt_path, transform=trainTransform)
test_data = MyDataset(txt_path=test_txt_path, transform=testTransform)

train_loader = DataLoader(
    dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size)

# 混淆矩阵
def plot_cm(labels, pre, savepath):
    conf_numpy = confusion_matrix(labels, pre)
    conf_numpy = conf_numpy.astype('float') / conf_numpy.sum(axis=1)
    conf_numpy_norm = np.around(conf_numpy, decimals=3)
    # conf_df = pd.DataFrame(conf_numpy)#将data和all_label_names制成DataFrame

    plt.figure(figsize=(8, 7))
    sns.heatmap(conf_numpy_norm, annot=True, cmap="Blues")  # 将data绘制为混淆矩阵
    plt.title('confusion matrix', fontsize=15)
    plt.ylabel('True labels', fontsize=14)
    plt.xlabel('Predict labels', fontsize=14)
    # plt.savefig(Path_All + 'Output/Train_ConfMatrix.png')
    plt.savefig(savepath)

import itertools
def plot_confusion_matrix(labels, pre, classes, savepath, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues,fontsize=20):
    conf_numpy = confusion_matrix(labels, pre)
    if normalize:
        conf_numpy = conf_numpy.astype('float') / conf_numpy.sum(axis = 1)
        conf_numpy = np.around(conf_numpy,decimals=3)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(conf_numpy)

    plt.figure(figsize=(8, 7))
    plt.imshow(conf_numpy, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=fontsize)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=fontsize)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=fontsize)
    plt.yticks(tick_marks, classes, fontsize=fontsize)

    fmt = '.3f' if normalize else 'd'
    thresh = conf_numpy.max() / 2.
    for i, j in itertools.product(range(conf_numpy.shape[0]), range(conf_numpy.shape[1])):
        plt.text(j, i, format(conf_numpy[i, j], fmt),
                 horizontalalignment="center",
                 fontsize=fontsize,
                 color="white" if conf_numpy[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=fontsize)
    plt.xlabel('Predicted label', fontsize=fontsize)
    plt.tight_layout()
    # plt.savefig(savepath+'.mpl')
    plt.savefig(savepath+'.png')
    plt.savefig(savepath+'.eps')

# 结合网页上的方法，修改训练函数
model = resnet18(pretrained=True).to(device)

class fc_part(nn.Module):
        # fc 全连接层
        def __init__(self):
            super().__init__()
            # self.fc1 = nn.Linear(512,3)
            self.fc1 = nn.Linear(512, 256)
            self.fc2 = nn.Linear(256, 120)
            self.fc3 = nn.Linear(120,5)

        def forward(self, x):
            x = nf.relu(self.fc1(x))
            x = nf.relu(self.fc2(x))
            x = nf.relu(self.fc3(x))
            # x = self.fc1(x)
            return x

model.fc = fc_part().to(device)
model.load_state_dict(torch.load('./model1_83/best_resnet18_model.pth'))

ignored_params = list(map(id, model.fc.parameters()))
base_params = filter(lambda p: id(
    p) not in ignored_params, model.parameters())
optimizer = optim.SGD([
    {'params': base_params},
    {'params': model.fc.parameters(), 'lr': lr*10}], lr, momentum=0.9, weight_decay=1e-4)
# 优化函数，随机梯度下降


# comment = f'batch_size{batch_size} lr{lr}'
if os.path.exists(log_dir):
    import shutil
    shutil.rmtree(log_dir)
tb = SummaryWriter(log_dir=log_dir)

# 模型训练+测试
def train_gesture(model, train_loader, test_loader, optimizer, device, num_epoch):
    print("training on ", device)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.1)
    test_epoch_list = []
    test_acc_list = []
    # 设置学习率下降策略 StepLR 固定学习率下降
    # 这里的optimizer函数不一样，不知道行不行，试一下先
    best_pred_list_total = []
    best_epoch = 0
    best_test_accuracy = torch.Tensor([0.0])  # 初始化最佳准确率
    for epoch in range(1, num_epoch+1):
        print(epoch, '\n')
        train_loss = 0.0
        train_correct = 0.0
        train_total = 0.0
        predicted_list = []
        labels_list = []
        scheduler.step()
        # 更新学习率
        with tqdm(train_loader, desc='Train') as t:
            # with函数 以t来代表tqdm中的东西
            model.train()
            for data in t:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                # 获取图片和标签
                # 常规操作，forward
                optimizer.zero_grad()
                # 清空梯度
                outputs = model.forward(inputs)
                l = nf.cross_entropy(outputs, labels)
                l.backward()
                optimizer.step()

                # 统计预测信息
                _, predicted = torch.max(outputs, axis=1)
                # print(predicted)
                # print(labels)
                train_total += labels.size(0)
                train_correct += torch.sum(predicted == labels).item()
                train_loss += l.item()

                # 设置进度条右边的显示信息
                t.set_postfix(train_loss=l.item(),
                                train_accuracy=train_correct/train_total)
        test_loss = 0.0
        test_correct = 0.0
        test_total = 0.0
        # pred_list_total = []
        # labels_list_total = []

        with torch.no_grad():
            model.eval()
            with tqdm(test_loader, desc='Test') as t:
                for data in t:
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model.forward(inputs)
                    l = nf.cross_entropy(outputs, labels)
                    test_loss += l.item()

                    _, predicted = torch.max(outputs, axis=1)

                    predicted_list.append(predicted)
                    labels_list.append(labels)

                    test_total += labels.size(0)
                    test_correct += torch.sum(predicted == labels).item()
                    t.set_postfix(test_loss=l.item(),
                                    test_accuracy=test_correct / test_total)   
        tb.add_scalar('train_loss', train_loss/train_total, epoch)
        tb.add_scalar('train_accuracy', train_correct/train_total, epoch)
        tb.add_scalar('test_loss', test_loss/test_total, epoch)
        tb.add_scalar('test_accuracy', test_correct/test_total, epoch)
        test_acc_each_epoch = test_correct/test_total
        test_epoch_list.append(epoch)
        test_acc_list.append(test_acc_each_epoch)
        test_acc_each_epoch = 0
        # predicted_list = torch.reshape([predicted_list,-1])
        predicted_list = [aa.tolist() for aa in predicted_list]
        # print(predicted_list,type(predicted_list))
        pred_list_total = [i for item in predicted_list for i in item]
        labels_list = [aa.tolist() for aa in labels_list]
        # labels_list = torch.tensor(labels_list)
        labels_list_total = [i for item in labels_list for i in item]
        # labels_list = torch.reshape([labels_list,-1])
#  print('epoch %d, train acc %.3f,test acc %.3f, time %.1f sec'%(epoch,train_correct / train_total, test_correct/test_total,time.time()-start))
        train_acc_sum = torch.tensor([round(train_correct/train_total, 5)])
        test_acc = torch.tensor([test_correct/test_total])
        if epoch == 1:
            train_acc_all = train_acc_sum
            test_acc_all = test_acc
            # print(train_acc_all,type(test_acc_all))
        else:
            train_acc_all = torch.cat([train_acc_all, train_acc_sum])
            test_acc_all = torch.cat([test_acc_all, test_acc])
            # print(train_acc_all,type(train_acc_all))
        # 最佳epoch的模型    
        if test_acc > best_test_accuracy:
            best_test_accuracy = test_acc
            best_epoch = epoch
            best_pred_list_total = pred_list_total.copy()
            torch.save(model.state_dict(), Path_All+'best_resnet18_model.pth')
        tb.add_scalar('best accuracy',best_test_accuracy.numpy(),epoch)
        tb.add_scalar('best_epoch',best_epoch,epoch)
        print('best accuracy: {}, best epoch: {}'.format(best_test_accuracy.numpy(),best_epoch))


    # plot_cm(labels_list_total, pred_list_total,Path_All + 'Output/Train_ConfMatrix.png')
    # plot_cm(labels_list_total, best_pred_list_total,Path_All + 'Output/Best_Train_ConfMatrix.png')
    plot_confusion_matrix(labels_list_total,pred_list_total,['Pushing & \nPulling','Beckoning','Rub \nFingers',"Plugging","Scaling"],Path_All + 'Output/Train_ConfMatrix',normalize=True)
    plot_confusion_matrix(labels_list_total,best_pred_list_total,['Pushing & \nPulling','Beckoning','Rub \nFingers',"Plugging","Scaling"],Path_All + 'Output/Best_Train_ConfMatrix',normalize=True)
    # print('best accuracy: {}, best epoch: {}'.format(best_test_accuracy.numpy(),best_epoch))

    train_acc_all = train_acc_all.tolist()
    test_acc_all = test_acc_all.tolist()

    # print(train_acc_all,type(test_acc_all))
    # 最好转换成列表，如果使tensor 可能会报错
    # semilogy(range(1,num_epochs+1),train_acc_all,'epochs','loss')
    # d2l.semilogy(range(1,num_epoch+1),test_acc_all,'epochs','acc',range(1,num_epoch+1),train_acc_all,['test','train'])
    plt.figure()
    d2l.semilogy(range(1, num_epoch+1), test_acc_all, 'epochs', 'acc')
    name = ['train_acc', 'test_acc']
    acc_file = pd.DataFrame(index=name, data=(train_acc_all, test_acc_all))
    acc_file.to_csv(Path_All + '/Output/test pred.csv', encoding='gbk')
    plt.show()
    # 等两次完整的迭代进行完毕后，保存训练好的模型及其参数
    torch.save(model.state_dict(), Path_All+'resnet18_model.pth')  # 保存模型的状态字典，也就是保存模型的参数信息
    # 将测试时的每个epoch的准确率信息写入一个.csv文件中
    # 打开一个.csv文件进行写入
    with open(Output_path + '/test_acc.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
    # 逐行写入数据
        for i in range(len(test_epoch_list)):
            row = [test_epoch_list[i], test_acc_list[i]]
            writer.writerow(row)



if __name__ == '__main__':
    train_gesture(model, train_loader, test_loader,
                    optimizer, device, num_epoch)

