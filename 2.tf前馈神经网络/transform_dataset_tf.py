# coding: utf-8
import numpy as np
import pickle
from two_layer_net import TwoLayerNet

from vocab import *
'''
将文本从词的序列转换为词语tf值的序列。每一个词语有两个tf值，
一个是在标签为positive的数据集中的tf值，另一个是在标签为
negative的数据集中的tf值。由于使用的神经网络为前馈神经网络
，因此虽然每个词语tf值对应一个二维向量，但是拉平为2个数据值。
因此处理后的数据集中，每个句子的长度为原来的两倍。
'''
with open("tf_pos.pkl","rb") as f:
    worddic_tf_pos = pickle.load(f)
with open("tf_neg.pkl","rb") as f:
    worddic_tf_neg = pickle.load(f)


def trans_n2tf(set):
    res=[]
    for sentence in set:
        tem=[]
        for num in sentence:
            tem.append(worddic_tf_pos[num])
            tem.append(worddic_tf_neg[num]*(-1))
        res.append(tem)
    return res

with open("train_set_w2n.pkl", "rb") as f:
    train_set = pickle.load(f)
with open("validate_set_w2n.pkl", "rb") as f:
    validate_set = pickle.load(f)
with open("test_set_w2n.pkl", "rb") as f:
    test_set = pickle.load(f)

train_set=trans_n2tf(train_set)
validate_set=trans_n2tf(validate_set)
test_set=trans_n2tf(test_set)

with open("train_set_tf.pkl", "wb") as f:
    pickle.dump(train_set,f)
with open("validate_set_tf.pkl", "wb") as f:
    pickle.dump(validate_set,f)
with open("test_set_tf.pkl", "wb") as f:
    pickle.dump(test_set,f)