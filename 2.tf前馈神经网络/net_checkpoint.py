# coding: utf-8
import numpy as np
import pickle
from two_layer_net import TwoLayerNet
from layers import *
'''
    checkpoint
    读取最终训练模型，并在测试集上展示模型最终训练的结果
'''
# if(len(validate_acc_list)==25):
#     with open('save_tf_net.pkl', 'wb') as f:
#         pickle.dump(network,f)
#         break
with open('save_tf_net.pkl','rb') as f:
    network=pickle.load(f)

with open("test_label.pkl","rb") as f:
    t_test = pickle.load(f)
with open("test_set_tf.pkl", "rb") as f:
    x_test = pickle.load(f)
x_test=np.array(x_test)
t_test=np.array(t_test)

test_acc = network.accuracy(x_test, t_test)
test_rec = network.recall(x_test, t_test, 1)
test_prec = network.precision(x_test, t_test, 1)

print("Performance in test set:")
print(f"   accuracy:{format(test_acc, '.4f')}  recall:{format(test_rec, '.4f')}   precision:{format(test_prec, '.4f')}")
