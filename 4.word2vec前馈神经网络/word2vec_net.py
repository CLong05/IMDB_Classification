# coding: utf-8
import numpy as np
import pickle
from two_layer_net import TwoLayerNet

# 读入数据
with open("train_set_w2v.pkl", "rb") as f:
    x_train = pickle.load(f)
with open("validate_set_w2v.pkl", "rb") as f:
    x_validate = pickle.load(f)
with open("test_set_w2v.pkl", "rb") as f:
    x_test = pickle.load(f)
with open("train_label.pkl","rb") as f:
    t_train = pickle.load(f)
with open("validate_label.pkl","rb") as f:
    t_validate = pickle.load(f)
with open("test_label.pkl","rb") as f:
    t_test = pickle.load(f)

x_train=np.array(x_train)
x_validate=np.array(x_validate)
x_test=np.array(x_test)
t_test=np.array(t_test)
t_train=np.array(t_train)
t_validate=np.array(t_validate)

network = TwoLayerNet(input_size=1000, hidden_size=200, output_size=2)

earlystop=15
iters_num = 100000
train_size = x_train.shape[0]
batch_size = 200
learning_rate = 0.025

train_loss_list = []
train_acc_list = []
validate_acc_list = []
test_acc_list=[]
test_rec_list=[]
test_prec_list=[]

# train_rec_list = []
# validate_rec_list = []
# train_prec_list = []
# validate_prec_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 梯度
    grad = network.gradient(x_batch, t_batch)

    # 更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        validate_acc = network.accuracy(x_validate, t_validate)
        # train_rec = network.recall(x_train, t_train,1)
        # validate_rec = network.recall(x_validate, t_validate,1)
        # train_prec = network.precision(x_train, t_train, 1)
        # validate_prec = network.precision(x_validate, t_validate, 1)

        train_acc_list.append(train_acc)
        validate_acc_list.append(validate_acc)

        test_acc = network.accuracy(x_test, t_test)
        test_rec = network.recall(x_test, t_test, 1)
        test_prec = network.precision(x_test, t_test, 1)
        test_acc_list.append(test_acc)
        test_rec_list.append(test_rec)
        test_prec_list.append(test_prec)
        # train_rec_list.append(train_rec)
        # validate_rec_list.append(validate_rec)
        # train_prec_list.append(train_prec)
        # validate_prec_list.append(validate_prec)

        print(f"epoch {len(train_acc_list)}:")
        print(f"  train_acc: {format(train_acc,'.4f')}    validate_acc: {format(validate_acc,'.4f')} ")

        if(len(validate_acc_list)>earlystop):  # 如果性能相较于earlystop个epoch前没有提升，则停止训练
            if(validate_acc<validate_acc_list[earlystop*(-1)]):
                max_index=0
                for i in range(len(validate_acc_list)):
                    if validate_acc_list[i] > validate_acc_list[max_index]:
                        max_index = i

                test_acc = test_acc_list[max_index]
                test_rec = test_rec_list[max_index]
                test_prec = test_prec_list[max_index]
                print("Performance in test set:")
                print(f"   accuracy:{format(test_acc,'.4f')}  recall:{format(test_rec,'.4f')}   precision:{format(test_prec,'.4f')}")
                break