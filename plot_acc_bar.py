import sys
import csv
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 存储六个列表的数据
x_values_list = []
y_values_list = []
count = 1
path_all = ['./model1/Output/test_acc_alone.csv','./model2/Output/test_acc_alone.csv','./model3/Output/test_acc_alone.csv',\
'./model4/Output/test_acc_alone.csv','./model5/Output/test_acc_alone.csv','./model6/Output/test_acc_alone.csv']

def plot_acc_bar(count):
    # 读取.csv文件中的数据并存储到对应的列表中
    for path in path_all:
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
            last_row = rows[-1]
            x_values_list.append("Scheme " + str(count))  # 添加空格
            y_values_list.append(float(last_row[1])*100)
        count += 1

    # 使用Seaborn绘制图形
    # sns.set(style="whitegrid")
    sns.set(style="ticks")
    plt.figure(figsize=(12,5))
    ax = sns.barplot(x=x_values_list, y=y_values_list, palette="pastel", ci=None, capsize=0.2, errwidth=2, width=0.5)  # 调整柱子宽度为0.5
    plt.xlabel('Scheme', fontsize=24)  # 添加x轴标签和字体大小
    plt.ylabel('Accuracy (%)', fontsize=24)  # 添加y轴标签和字体大小
    # plt.title('Recognition Accuracy for Different Models', fontsize=24)  # 添加标题和字体大小
    plt.xticks(fontsize=24, rotation=0)  # 调整x轴标签字体大小和角度
    plt.yticks(fontsize=24)  # 调整y轴标签字体大小
    plt.tight_layout()  # 调整布局，防止标签被裁剪

    # 在柱子上显示数值
    for index, value in enumerate(y_values_list):
        ax.text(index, value, f'{value:.1f}%', ha='center', va='bottom', fontsize=24)

    # 添加图例
    # plt.legend(['Test Accuracy'], loc='upper left', fontsize=12)

    # 隐藏右边和上边的边框
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # # 显示水平虚线网格
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig('./all_model_test_acc_bar.png', dpi=300)  # 保存图像并设置分辨率
    plt.savefig('./all_model_test_acc_bar.eps', format='eps')  # 保存eps格式图像
    plt.show()

if __name__ == "__main__":
    plot_acc_bar(count)
