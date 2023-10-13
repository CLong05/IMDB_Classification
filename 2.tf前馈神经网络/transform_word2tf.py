# coding: utf-8
import numpy as np
import pickle
from two_layer_net import TwoLayerNet
import math


from vocab import *

# 读入清洗后的数据
with open("train_set_remove_sw.pkl", "rb") as f:
    train_set = pickle.load(f)
with open("validate_set_remove_sw.pkl", "rb") as f:
    validate_set = pickle.load(f)
with open("test_set_remove_sw.pkl", "rb") as f:
    test_set = pickle.load(f)
with open("train_label.pkl","rb") as f:
    train_label = pickle.load(f)
with open("validate_label.pkl","rb") as f:
    validate_label = pickle.load(f)
with open("test_label.pkl","rb") as f:
    test_label = pickle.load(f)

# 统计词频，构造词典，并将文本序列化
sentences = train_set + validate_set
labels = train_label + validate_label
ws = Vocab()
for sentence in sentences:
    # 统计词频
    ws.fit(sentence)

# 构造词典
ws.build_vocab(min_count=100)
print(len(ws.dict))
print(ws.dict)

# 把句子转换成数字序列
def transform_w2n(set):
    set_w2n = []
    for sent in set:
        res = ws.transform(sent, max_len=100)
        set_w2n.append(res)
    return set_w2n


train_set_w2n = transform_w2n(train_set)
validate_set_w2n = transform_w2n(validate_set)
test_set_w2n = transform_w2n(test_set)

with open("train_set_w2n.pkl", "wb") as f:
    pickle.dump(train_set_w2n, f)
with open("validate_set_w2n.pkl", "wb") as f:
    pickle.dump(validate_set_w2n, f)
with open("test_set_w2n.pkl", "wb") as f:
    pickle.dump(test_set_w2n, f)


with open("train_set_w2n.pkl", "rb") as f:
    train_set = pickle.load(f)
with open("validate_set_w2n.pkl", "rb") as f:
    validate_set = pickle.load(f)
with open("test_set_w2n.pkl", "rb") as f:
    test_set = pickle.load(f)
with open("train_label.pkl","rb") as f:
    train_label = pickle.load(f)
with open("validate_label.pkl","rb") as f:
    validate_label = pickle.load(f)
with open("test_label.pkl","rb") as f:
    test_label = pickle.load(f)

sentences = train_set + validate_set
labels = train_label + validate_label

# 统计各个词的词频
worddic_tf_pos = {}  # 统计词频
worddic_tf_neg = {}  # 统计词频
for i in range(len(ws.dict)):
    worddic_tf_pos[i] = 0
    worddic_tf_neg[i] = 0

for i in range(len(sentences)):
    if labels[i] == 1:
        for num in sentences[i]:
            worddic_tf_pos[num] = worddic_tf_pos.get(num,0)+1
    if labels[i] == 0:
        for num in sentences[i]:
            worddic_tf_neg[num] = worddic_tf_neg.get(num,0)+1

# 将未匹配与填充符号的词频设置为0
for i in range(2):
    worddic_tf_neg[i]=0
    worddic_tf_pos[i]=0

# 计算各个词的tf值
for i in range(len(worddic_tf_pos)):
    worddic_tf_pos[i]=math.log10( worddic_tf_pos[i] + 1 )
    worddic_tf_neg[i]=math.log10( worddic_tf_neg[i] + 1 )

with open("tf_pos.pkl","wb") as f:
    pickle.dump(worddic_tf_pos,f)
with open("tf_neg.pkl","wb") as f:
    pickle.dump(worddic_tf_neg,f)

#显示结果
print(worddic_tf_pos)
print(worddic_tf_neg)


# 将文本从词的序列转换为索引tf值的序列
def trans_n2tf(set):
    res=[]
    for sentence in set:
        tem=[]
        for num in sentence:
            tem.append(worddic_tf_pos[num])
            tem.append(worddic_tf_neg[num]*(-1))
        res.append(tem)
    return res


train_set=trans_n2tf(train_set)
validate_set=trans_n2tf(validate_set)
test_set=trans_n2tf(test_set)

with open("train_set_tf.pkl", "wb") as f:
    pickle.dump(train_set,f)
with open("validate_set_tf.pkl", "wb") as f:
    pickle.dump(validate_set,f)
with open("test_set_tf.pkl", "wb") as f:
    pickle.dump(test_set,f)

# print(train_set)