from __future__ import print_function
from lib2to3.pgen2.grammar import opmap_raw
import torch
from torch import nn,optim, tensor
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import time
import torch.nn.functional as nf
import csv
import warnings
warnings.filterwarnings(action='ignore')

import glob #返回文件路径中所有匹配的文件名
from PIL import Image ,ImageStat #图像处理的库
from tqdm import tqdm #进度条库
import cv2 #open cv
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18
#调用pytorch中的resnet18网络进行预训练权重

from torch.utils.tensorboard import SummaryWriter

###混淆矩阵
from sklearn.metrics import confusion_matrix
import seaborn as sns

Path_All = './model1/'
train_path = Path_All + '/train/'
test_path = Path_All + 'test/'
test_txt_path = Path_All + 'test.txt'
batch_size = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_all = {}
test_all = {}
for trainable in [True,False]:
    if trainable:
        for category in ['pic1','pic2','pic3','pic4','pic5']:
            train_all[category] = glob.glob(os.path.join(train_path + category,'*.jpg'))   #这种写法是分了两个文件夹
            # print(os.path.join(train_path + category,'*.jpg'))
    else:
        for category in ['pic1','pic2','pic3','pic4','pic5']:
            test_all[category] = glob.glob(os.path.join(test_path + category,'*.jpg'))
#结果显示有几个图片尺寸与其他的不符,手动删除它
###制作数据集对应的标签


class MyDataset(Dataset):
    def __init__(self,txt_path,transform = None , target_transform = None):
        fh = open(txt_path,'r')
        imgs = []
        for line in fh:
            line = line.rstrip()#删除字符串末尾的指定字符，默认为空格
            words = line.split()#分割字符串，默认以空格为分隔符进行全分割
            img_path = words[0]
            img_path = img_path.replace('\\','/') # linux中路径“\\”无法工作
            img_label = int(words[1])
            # print(words[0],'\n',words[1],type(words))
            imgs.append((img_path,img_label))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self,index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        #convert将图像转化为对应的像素值，有三种模式1、RGB，三通道。2、1,01模式，3、L模式，将每个像素点转化为0-255，单通道
        if self.transform is not None:
            img = self.transform(img)

        return img, label
    def __len__(self):
        return len(self.imgs)

train_paths = train_all['pic1'] + train_all['pic2'] + train_all['pic3']+ train_all['pic4']+ train_all['pic5']
m_list, s_list = [],[]
print('Read pic: \n')
for path in tqdm(train_paths):  #tqdm进度条模式，挺有意思的
    img = cv2.imread(path)
    img = img / 255.0
    m,s = cv2.meanStdDev(img)
    m_list.append(m.reshape((3,)))
    s_list.append(s.reshape((3,)))
m_array = np.array(m_list)
s_array = np.array(s_list)
#转换为矩阵
m = m_array.mean(axis = 0, keepdims = True)
s = s_array.mean(axis = 0, keepdims = True)

# print(m[0][::-1])
# print(s[0][::-1])
###数据预处理，编写transform 进行数据增强
normMean = m[0][::-1].copy()
normStd = s[0][::-1].copy()

normTransform = transforms.Normalize(normMean,normStd)
testTransform = transforms.Compose([
    #transforms.Resize((280,280)),
    # transforms.CenterCrop((1900,990)),
    # transforms.Resize((40,30)),
    transforms.ToTensor(),
    normTransform,
])

test_data = MyDataset(txt_path=test_txt_path,transform = testTransform)
test_loader = DataLoader(dataset = test_data,batch_size = batch_size)

new_model = resnet18(pretrained=True).to(device)

class fc_part(nn.Module):
        # fc 全连接层
        def __init__(self):
            super().__init__()
            # self.fc1 = nn.Linear(512,512)
            self.fc1 = nn.Linear(512, 256)
            self.fc2 = nn.Linear(256, 120)
            self.fc3 = nn.Linear(120,5)

        def forward(self, x):
            x = nf.relu(self.fc1(x))
            x = nf.relu(self.fc2(x))
            x = nf.relu(self.fc3(x))
            # x = self.fc1(x)
            return x
new_model.fc = fc_part().to(device)
new_model.load_state_dict(torch.load(Path_All+'best_resnet18_model.pth'))


# def plot_cm(labels,pre):
#     conf_numpy = confusion_matrix(labels,pre)
#     # print(conf_numpy,type(conf_numpy))
#     conf_numpy = conf_numpy.astype('float') / conf_numpy.sum(axis = 1)
#     conf_numpy_norm = np.around(conf_numpy,decimals=3)
#     # conf_df = pd.DataFrame(conf_numpy)#将data和all_label_names制成DataFrame

#     plt.figure(1,figsize=(8,7))
#     # sns.heatmap(conf_numpy_norm,annot=True,fmt="d",cmap="BuPu")#将data绘制为混淆矩阵
#     sns.heatmap(conf_numpy_norm,annot=True,cmap="Blues")#将data绘制为混淆矩阵
#     plt.title('confusion matrix',fontsize = 15)
#     plt.ylabel('True labels',fontsize = 14)
#     plt.xlabel('Predict labels',fontsize = 14)
#     plt.tight_layout()
#     plt.savefig(Path_All + 'Output/Test_ConfMatrix.png')
#     plt.savefig(Path_All + 'Output/Test_ConfMatrix.eps',format='eps')

# import itertools
# def plot_confusion_matrix(labels, pre, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues,fontsize=28):
#     conf_numpy = confusion_matrix(labels, pre)
#     if normalize:
#         conf_numpy = conf_numpy.astype('float') / conf_numpy.sum(axis = 1)
#         conf_numpy = np.around(conf_numpy,decimals=3)
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#     print(conf_numpy)

#     plt.figure(1, figsize=(8, 7))
#     plt.imshow(conf_numpy, interpolation='nearest', cmap=cmap)
#     plt.title(title, fontsize=fontsize)
#     cbar = plt.colorbar()
#     cbar.ax.tick_params(labelsize=fontsize)

#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45, fontsize=fontsize)
#     plt.yticks(tick_marks, classes, fontsize=fontsize)

#     fmt = '.3f' if normalize else 'd'
#     thresh = conf_numpy.max() / 2.
#     for i, j in itertools.product(range(conf_numpy.shape[0]), range(conf_numpy.shape[1])):
#         plt.text(j, i, format(conf_numpy[i, j], fmt),
#                  horizontalalignment="center",
#                  fontsize=fontsize+10,
#                  color="white" if conf_numpy[i, j] > thresh else "black")

#     plt.ylabel('True label', fontsize=fontsize)
#     plt.xlabel('Predicted label', fontsize=fontsize)
#     plt.tight_layout()
#     plt.savefig(Path_All + 'Output/Test_ConfMatrix.png')
#     plt.savefig(Path_All + 'Output/Test_ConfMatrix.eps', format='eps')


# def plot_confusion_matrix_v2(conf_numpy, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues,fontsize=28):
#     if normalize:
#         conf_numpy = conf_numpy.astype('float') / conf_numpy.sum(axis = 1)
#         conf_numpy = np.around(conf_numpy,decimals=3)
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#     print(conf_numpy)

#     plt.figure(1, figsize=(8, 7))
#     plt.imshow(conf_numpy, interpolation='nearest', cmap=cmap)
#     plt.title(title, fontsize=fontsize)
#     cbar = plt.colorbar()
#     cbar.ax.tick_params(labelsize=fontsize)

#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45, fontsize=fontsize)
#     plt.yticks(tick_marks, classes, fontsize=fontsize)

#     fmt = '.3f' if normalize else 'd'
#     thresh = conf_numpy.max() / 2.
#     for i, j in itertools.product(range(conf_numpy.shape[0]), range(conf_numpy.shape[1])):
#         plt.text(j, i, format(conf_numpy[i, j], fmt),
#                  horizontalalignment="center",
#                  fontsize=fontsize+10,
#                  color="white" if conf_numpy[i, j] > thresh else "black")

#     plt.ylabel('True label', fontsize=fontsize)
#     plt.xlabel('Predicted label', fontsize=fontsize)
#     plt.tight_layout()
#     plt.savefig(Path_All + 'Output/Test_ConfMatrix.png')
#     plt.savefig(Path_All + 'Output/Test_ConfMatrix.eps', format='eps')

import itertools
def plot_confusion_matrix(labels, pre, classes, savepath, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues,fontsize=22):
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


def predict_gesture(model,test_loader,device):
    predicted_list,labels_list = [],[]
    test_loss = 0.0
    test_correct = 0.0
    test_total = 0.0
    with torch.no_grad():
        model.eval()
        with tqdm(test_loader,desc = 'Test') as t:
            for data in t :
                inputs ,labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model.forward(inputs)
                l = nf.cross_entropy(outputs,labels)
                test_loss += l.item()
                _,predicted = torch.max(outputs,axis = 1)
                predicted_list.append(predicted)
                labels_list.append(labels)
                test_total += labels.size(0)
                test_correct += torch.sum(predicted == labels).item()
                # print(labels,'\n',predicted)
    test_acc = test_correct/test_total
    print("Accuracy: {}".format(test_acc))
    # 打开一个.csv文件进行写入
    with open(Path_All + 'Output/test_acc_alone.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
    # 逐行写入数据
        row = [1, test_acc]
        writer.writerow(row)
    predicted_list=[aa.tolist() for aa in predicted_list]
    # print(predicted_list,type(predicted_list))
    pred_list_total = [i for item in predicted_list for i in item]
    labels_list=[aa.tolist() for aa in labels_list]
    # labels_list = torch.tensor(labels_list)
    labels_list_total = [i for item in labels_list for i in item]
    # plot_cm(labels_list_total,pred_list_total)
    plot_confusion_matrix(labels=labels_list_total,pre=pred_list_total,classes=['Pushing & \nPulling','Beckoning','Rubbing \nFingers',"Plugging","Scaling"],normalize=True,savepath=Path_All + 'Output/Test_ConfMatrix')
    plt.show()

if __name__ == '__main__':
    predict_gesture(new_model,test_loader,device)
    # conf_numpy_model1 = np.array([[1.,0.,0.],[0.02,0.88,0.1],[0.025,0.125,0.85]])
    # plot_confusion_matrix_v2(conf_numpy_model1,classes=['Pushing & \nPulling','Beckoning','Rubbing \nFingers'],normalize=True)

