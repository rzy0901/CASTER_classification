"""
依次读取每一个model文件夹中的test_acc.csv文件并使用matplotlib绘制测试分类准确率随epoch变化的曲线
"""
import sys
import csv
import os
import matplotlib.pyplot as plt

# 存储六个列表的数据
x_values_list = []
y_values_list = []

path_all = ['./model1/Output/test_acc.csv','./model2/Output/test_acc.csv','./model3/Output/test_acc.csv',\
'./model4/Output/test_acc.csv','./model5/Output/test_acc.csv','./model6/Output/test_acc.csv']
def plot_acc():
    # 读取.csv文件中的数据并存储到对应的列表中
    for path in path_all:
        x_values = []
        y_values = []

        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                x_values.append(float(row[0]))
                y_values.append(float(row[1]))
        x_values_list.append(x_values)
        y_values_list.append(y_values)
    # print(x_values_list)
   # 使用matplotlib绘制图形
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink']
    for i in range(6):
        plt.plot(x_values_list[i], y_values_list[i], color=colors[i])

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Epoch Versus Accuracy for different Model')
    plt.legend(['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5', 'Model 6'])
    plt.savefig('./all_model_test_acc.eps',format='eps')
    plt.savefig('./all_model_test_acc.png')
    plt.show()



if __name__ == "__main__":
    plot_acc()
