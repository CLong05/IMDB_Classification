import torch
import pandas as pd
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import pickle
import re   #正则表达库
import zipfile
from torch.utils.data import Dataset, DataLoader

'''
进行数据的清洗，原始数据中包含了HTML标签和URL等与文本情感无关的噪音
同时统一单词大小写、去除标点符号，并进行分词处理
修改标签，positive改为1，negative改为0
将处理后的三个数据集以及标签保存为pkl文件
'''

# 读取数据集
file = pd.read_csv(r'D:\.A.大三上学习内容\NLP\Mid-term project\IMDB_sentiment using tf/IMDB Dataset.csv', sep=',')

# 获取标签并修改
labels = file._get_column_array(1)
labels = [1 if i == 'positive' else 0 for i in labels]

# 取出文本并分割为训练集和测试集
data = file._get_column_array(0)
train_data = data[:30000]
train_label = labels[:30000]
validate_data = data[30000:40000]
validate_label = labels[30000:40000]
test_data = data[40000:]
test_label = labels[40000:]


# 删除标点符号，HTML标签和URL等与文本情感无关的噪音
def tokenlize(sentence):
    """
    进行文本分词
    :param sentence: str
    :return: [str,str,str]
    """
    fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
                '\?', '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”',
                '“' ]
    sentence = sentence.lower()  # 把大写转化为小写
    sentence = re.sub("<br />", " ", sentence)
    sentence = re.sub("|".join(fileters), " ", sentence)    #正则表达，代换filters的符号
    result = [i for i in sentence.split(" ") if len(i) > 0]

    return result


# 数据预处理
train_set=[]
for sentence in train_data:
    res = [tokenlize(sentence)]
    train_set+=res

validate_set=[]
for sentence in validate_data:
    res = [tokenlize(sentence)]
    validate_set+=res

test_set=[]
for sentence in test_data:
    res = [tokenlize(sentence)]
    test_set+=res

#保存处理好的数据
with open("train_set.pkl", "wb") as f:
    pickle.dump(train_set, f)
with open("train_label.pkl", "wb") as f:
    pickle.dump(train_label, f)
with open("validate_set.pkl", "wb") as f:
    pickle.dump(validate_set, f)
with open("validate_label.pkl", "wb") as f:
    pickle.dump(validate_label, f)
with open("test_set.pkl", "wb") as f:
    pickle.dump(test_set, f)
with open("test_label.pkl", "wb") as f:
    pickle.dump(test_label, f)

# #to load it
# with open("train.pkl", "rb") as f:
#     test_set = pickle.load(f)
# print(test_set)

