import os
import random

PATH_IMAGE = './data/Glomeruli-dataset/images'
all_file_list = os.listdir(PATH_IMAGE)
all_file_num = len(all_file_list)
random.shuffle(all_file_list) # 随机打乱全部数据文件名列表
train_ratio = 0.8
test_ratio = 1 - train_ratio
train_file_list = all_file_list[:int(all_file_num*train_ratio)]
test_file_list = all_file_list[int(all_file_num*train_ratio):]

print('数据集图像总数', all_file_num)
print('训练集划分比例', train_ratio)
print('训练集图像个数', len(train_file_list))
print('测试集图像个数', len(test_file_list))

with open('./data/Glomeruli-dataset/splits/train.txt', 'w') as f:
    f.writelines(line.split('.')[0] + '\n' for line in train_file_list)
with open('./data/Glomeruli-dataset/splits/val.txt', 'w') as f:
    f.writelines(line.split('.')[0] + '\n' for line in test_file_list)