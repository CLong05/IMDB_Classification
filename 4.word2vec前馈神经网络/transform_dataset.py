# coding: utf-8
import numpy as np
import pickle
from two_layer_net import TwoLayerNet
from gensim.models import word2vec
from vocab import *

# 导入模型
model = word2vec.Word2Vec.load('D:\.A.大三上学习内容\\NLP\Mid-term project\IMDB_sentiment using w2v\IMDB_word2vec.model')

# 读入清洗后的数据
with open("train_set_remove_sw.pkl", "rb") as f:
    train_set = pickle.load(f)
with open("validate_set_remove_sw.pkl", "rb") as f:
    validate_set = pickle.load(f)
with open("test_set_remove_sw.pkl", "rb") as f:
    test_set = pickle.load(f)
# with open("train_label.pkl","rb") as f:
#     train_label = pickle.load(f)
# with open("validate_label.pkl","rb") as f:
#     validate_label = pickle.load(f)
# with open("test_label.pkl","rb") as f:
#     test_label = pickle.load(f)

# 统计词频，构造词典，并将文本序列化
sentences = train_set + validate_set
ws = Vocab()
for sentence in sentences:
    # 统计词频
    ws.fit(sentence)

# 构造词典
ws.build_vocab(min_count=1)
#print(ws.dict)

# 建立 "单词：word2vec" 的字典，word2vec值取全部值
w2v_dic = {}
for word in ws.dict.keys():
    num = ws.dict[word]
    value=[0,0,0,0,0,0,0,0,0,0]
    if word in model.wv:
        value = model.wv[word]
    w2v_dic[num] = value

# #求出平均句长
# sum_len=0
# for sentence in train_set:
#     sum_len+=len(sentence)
# print(sum_len/len(train_set))
# #保存处理好的数据
# with open("num2value_dic.pkl", "wb") as f:
#     pickle.dump(w2v_dic, f)
# with open("word2num_dic.pkl", "wb") as f:
#     pickle.dump(ws.dict, f)

# 把句子转换成数字序列,在转化为对应的word2vec的值
def transform_w2v(set):
    set_w2v = []
    for sent in set:
        result=[]
        res1 = ws.transform(sent, max_len=100)
        for word_num in res1:
            word_val = w2v_dic[word_num]
            for val in word_val:
                result.append(val)
        set_w2v.append(result)
    return set_w2v

train_set_w2v = transform_w2v(train_set)
validate_set_w2v = transform_w2v(validate_set)
test_set_w2v = transform_w2v(test_set)

with open("train_set_w2v.pkl", "wb") as f:
    pickle.dump(train_set_w2v, f)
with open("validate_set_w2v.pkl", "wb") as f:
    pickle.dump(validate_set_w2v, f)
with open("test_set_w2v.pkl", "wb") as f:
    pickle.dump(test_set_w2v, f)

print(train_set_w2v[0])

# # 验证结果
# import pickle
# with open("train_set_w2v.pkl", "rb") as f:
#     train_set_w2v = pickle.load(f)
# for i in range(10):
#     print(train_set_w2v[i])