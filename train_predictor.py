import argparse
import math
import os
import time
from datetime import datetime
from functools import partial

# import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
# from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from collections import OrderedDict
import pandas as pd
import csv
import matplotlib.pyplot as plt

import rclstm
from rclstm import RNN, LSTMCell, RCLSTMCell
import time

np.random.seed(1000)

def get_args(parser):
    parser.add_argument('--data', default='F:/RCLSTM+CNN/pythonProject3/training set/ACET.csv', help='path to dataset')
    parser.add_argument('--model', default='lstm', choices=['lstm', 'rclstm'], help='the model to use')
    parser.add_argument('--connectivity', type=float, default=.5, help='the neural connectivity')
    parser.add_argument('--save', default='./model', help='The path to save model files')
    parser.add_argument('--hidden-size', type=int, default=200, help='The number of hidden units')
    parser.add_argument('--batch-size', type=int, default=32, help='The size of each batch')
    parser.add_argument('--input-size', type=int, default=1, help='The size of input data')
    parser.add_argument('--max-iter', type=int, default=2, help='The maximum iteration count')
    parser.add_argument('--gpu', default=True, action='store_true', help='The value specifying whether to use GPU')
    parser.add_argument('--time-window', type=int, default=60, help='The length of time window')
    parser.add_argument('--dropout', type=float, default=1., help='Dropout')
    parser.add_argument('--num-layers', type=int, default=1, help='The number of RNN layers')
    return parser

# get model's parameter
parser = argparse.ArgumentParser()
parser = get_args(parser)
args = parser.parse_args()
print(args)

data_path = args.data
model_name = args.model
# save_dir = args.save
hidden_size = args.hidden_size
batch_size = args.batch_size
max_iter = args.max_iter
use_gpu = args.gpu
# connectivity = args.connectivity
time_window = args.time_window
input_size = args.input_size
dropout = args.dropout
num_layers = args.num_layers

def shufflelists(X, Y):
    ri=np.random.permutation(len(X))
    X_shuffle = [X[i].tolist() for i in ri]
    Y_shuffle = [Y[i].tolist() for i in ri]
    return np.array(X_shuffle), np.array(Y_shuffle)

# load data
df = pd.read_csv("training set/ACET.csv")
df_a = pd.read_csv('training set/AAOI.csv')
df_b = pd.read_csv('training set/AAT.csv')

data = np.array(df['close'])
data_a = np.array(df_a['close'])
data_b = np.array(df_b['close'])

data = np.reshape(data, (len(data), 1))
data_a = np.reshape(data_a, (len(data_a), 1))
data_b = np.reshape(data_b, (len(data_b), 1))

# take the logarithm of the original data
new_data = []
for x in data:
    if x > 0:
        new_data.append(np.log10(x))
    else:
        new_data.append(0.001)

new_data_a = []
for x in data_a:
    if x > 0:
        new_data_a.append(np.log10(x))
    else:
        new_data_a.append(0.001)

new_data_b = []
for x in data_b:
    if x > 0:
        new_data_b.append(np.log10(x))
    else:
        new_data_b.append(0.001)

new_data = np.array(new_data)
new_data_a = np.array(new_data_a)
new_data_b = np.array(new_data_b)

# handle abnormal data
new_data = new_data[new_data>0]
data = new_data[new_data<6]

new_data_a = new_data_a[new_data_a>0]
data_a = new_data_a[new_data_a<6]

new_data_b = new_data_b[new_data_b>0]
data_b = new_data_b[new_data_b<6]
# min-max normalization
max_data = np.max(data)
min_data = np.min(data)
data = (data-min_data)/(max_data-min_data)

max_data_a = np.max(data_a)
min_data_a = np.min(data_a)
data_a = (data_a-min_data_a)/(max_data_a-min_data_a)

max_data_b = np.max(data_b)
min_data_b = np.min(data_b)
data_b = (data_b-min_data_b)/(max_data_b-min_data_b)

df = pd.DataFrame({'temp':data})
df_a = pd.DataFrame({'temp':data_a})
df_b = pd.DataFrame({'temp':data_b})

# define function for create N lags
def create_lags(df, N):
    for i in range(N):
        df['Lag' + str(i+1)] = df.temp.shift(i+1)
    return df

# create time-windows lags
df = create_lags(df,time_window)
df_a = create_lags(df_a,time_window)
df_b = create_lags(df_b,time_window)

# the first 1000 days will have missing values. can't use them.
df = df.dropna()
df_a = df_a.dropna()
df_b = df_b.dropna()

# create X and y
y = df.temp.values
X = df.iloc[:, 1:].values

y_a = df_a.temp.values
X_a = df_a.iloc[:, 1:].values

y_b = df_b.temp.values
X_b = df_b.iloc[:, 1:].values

# train on 80% of the data
train_idx = int(len(df) * .8)

# create train and test data
train_X, train_Y, test_X, test_Y = X[:train_idx], y[:train_idx], X[train_idx:], y[train_idx:]
test_X_a, test_Y_a = X_a, y_a
test_X_b, test_Y_b = X_b, y_b

print('the number of train data: ', len(train_X))
print('the number of test data: ', len(test_X))
print('the shape of input: ', train_X.shape)
print('the shape of target: ', train_Y.shape)

def compute_loss_accuracy(loss_fn, data, label):
    hx = None
    _, (h_n, _) = model[0](input_=data, hx=hx)
    logits = model[1](h_n[-1])
    loss = torch.sqrt(loss_fn(input=logits, target=label))
    return loss, logits

#learning rate decay
def exp_lr_scheduler(optimizer, epoch, init_lr=1e-2, lr_decay_epoch=3):
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))
    if epoch % lr_decay_epoch == 0:
        print("LR is set to {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

save_dir = args.save
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)
loss_fn = nn.MSELoss()
num_batch = int(math.ceil(len(train_X) // batch_size))
print('the number of batches: ', num_batch)

# train RCLSTM with different neural connection ratio
dic = {}
dic1 = {}
d1 = {}
d2 = {}

print(train_X.shape)

for connectivity in [0.01, 0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    print('neural connection ratio:', connectivity)
    if model_name in ['lstm', 'rclstm']:
        rnn_model = RNN(device=device, cell_class=model_name, input_size=input_size,
                        hidden_size=hidden_size, connectivity=connectivity,
                        num_layers=num_layers, batch_first=True, dropout=dropout)
    else:
        raise ValueError
    fc2 = nn.Linear(in_features=hidden_size, out_features=input_size)
    dropout_layer = nn.Dropout(dropout)
    conv1d_layer = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=5)
    model = nn.Sequential(OrderedDict([
            ('rnn', rnn_model),
            ('fc2', fc2),
            # ('dropout', dropout_layer),
            ('conv1d', conv1d_layer),
            # ('dropout', dropout_layer),
            ('fc2', fc2),
            ]))

    # if use_gpu:
    #     model.cuda()
    model.to(device)

    optim_method = optim.Adam(params=model.parameters())

    iter_cnt = 0
    while iter_cnt < max_iter:
        train_inputs, train_targets = shufflelists(train_X, train_Y)
        optimizer = exp_lr_scheduler(optim_method, iter_cnt, init_lr=0.01, lr_decay_epoch=3)
        for i in range(num_batch):
            low_index = batch_size*i
            high_index = batch_size*(i+1)
            if low_index <= len(train_inputs)-batch_size:
                batch_inputs = train_inputs[low_index:high_index].reshape(batch_size, time_window, 1).astype(np.float32)
                batch_targets = train_targets[low_index:high_index].reshape((batch_size, 1)).astype(np.float32)
            else:
                batch_inputs = train_inputs[low_index:].astype(float)
                batch_targets = train_targets[low_index:].astype(float)

            batch_inputs = torch.from_numpy(batch_inputs).to(device)
            batch_targets = torch.from_numpy(batch_targets).to(device)

            # if use_gpu:
            #     batch_inputs = batch_inputs.cuda()
            #     batch_targets = batch_targets.cuda()

            model.train(True)
            model.zero_grad()
            train_loss, logits = compute_loss_accuracy(loss_fn=loss_fn, data=batch_inputs, label=batch_targets)
            train_loss.backward()
            optimizer.step()

            if i % 20 == 0:
                print('the %dth iter, the %dth batch, train loss is %.4f' % (iter_cnt, i, train_loss.item()))

        # save model
        save_path = '{}/{}'.format(save_dir, int(round(connectivity/.01)))
        if os.path.exists(save_path):
            torch.save(model, os.path.join(save_path, str(iter_cnt)+'.pt'))
        else:
            os.makedirs(save_path)
            torch.save(model, os.path.join(save_path, str(iter_cnt)+'.pt'))
        iter_cnt += 1

    # define test loss function
    def compute_test_loss_accuracy(loss_fn, data, label):
        model_load = torch.load(os.path.join(save_path, str(iter_cnt-1) + '.pt'))
        hx = None
        _, (h_n, _) = model_load[0](input_=data, hx=hx)
        logits = model_load[1](h_n[-1])
        loss = torch.sqrt(loss_fn(input=logits, target=label))
        return loss, logits

    # compute test loss and average running time
    # tic = time.time()

    iterations = 300    # 重复计算的轮次

    random_input = torch.randn(input_size, hidden_size, 1).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # GPU预热
    for i in range(50):
        test_inputs, test_targets = test_X, test_Y

        test_inputs = test_inputs.reshape(len(test_inputs), -1, 1).astype(np.float32)
        test_targets = test_targets.reshape(len(test_targets), 1).astype(np.float32)

        test_inputs = torch.from_numpy(test_inputs).to(device)
        test_targets = torch.from_numpy(test_targets).to(device)

        test_loss, logits1 = compute_test_loss_accuracy(loss_fn=loss_fn, data=test_inputs, label=test_targets)

    # 测速
    times = torch.zeros(iterations)    # 储存每轮iteration的时间
    sum = 0
    for i in range(iterations):
        starter.record()
        test_inputs, test_targets = test_X, test_Y

        test_inputs = test_inputs.reshape(len(test_inputs),-1, 1).astype(np.float32)
        test_targets = test_targets.reshape(len(test_targets), 1).astype(np.float32)

        test_inputs = torch.from_numpy(test_inputs).to(device)
        test_targets = torch.from_numpy(test_targets).to(device)

        test_loss, logits1= compute_test_loss_accuracy(loss_fn=loss_fn, data=test_inputs, label=test_targets)
        ender.record()
        # 同步gpu时间
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        times[i] = curr_time
        sum += float(test_loss)
    # toc = time.time()

    mean_time = times.mean().item()

    test_loss_average = sum / iterations
    dic[f'{connectivity}'] = float(test_loss)
    dic1[f'{connectivity}'] = float(mean_time)

    print(f"Under connectivity: {connectivity}, the test loss on test set is {test_loss_average}, running time is {mean_time}s")
    print('\n')

    # the test loss on df_a(AAOI)
    test_inputs_a, test_targets_a = test_X_a, test_Y_a

    test_inputs_a = test_inputs_a.reshape(len(test_inputs_a), -1, 1).astype(np.float32)
    test_targets_a = test_targets_a.reshape(len(test_targets_a), 1).astype(np.float32)

    test_inputs_a = torch.from_numpy(test_inputs_a).to(device)
    test_targets_a = torch.from_numpy(test_targets_a).to(device)

    test_loss_a, logits1_a = compute_test_loss_accuracy(loss_fn=loss_fn, data=test_inputs_a, label=test_targets_a)

    print(f"Under connectivity: {connectivity}, the test loss on test_set_a(AAOI) is {test_loss_a}")

    # the test loss on df_b(AAT)
    test_inputs_b, test_targets_b = test_X_b, test_Y_b

    test_inputs_b = test_inputs_b.reshape(len(test_inputs_b), -1, 1).astype(np.float32)
    test_targets_b = test_targets_b.reshape(len(test_targets_b), 1).astype(np.float32)

    test_inputs_b = torch.from_numpy(test_inputs_b).to(device)
    test_targets_b = torch.from_numpy(test_targets_b).to(device)

    test_loss_b, logits1_b = compute_test_loss_accuracy(loss_fn=loss_fn, data=test_inputs_b, label=test_targets_b)

    print(f"Under connectivity: {connectivity}, the test loss on test_set_a(AAT) is {test_loss_b}")
    print('\n')

    d1[f'{connectivity}'] = float(test_loss_a)
    d2[f'{connectivity}'] = float(test_loss_b)

    # 在 测试集上 画 时间-预测值 和 时间-实际值 图像
    if connectivity == 0.01:
        logits1 = logits1.cpu().detach().numpy()
        logits1 = logits1.reshape(1, -1)
        logits1 = (logits1.tolist())[0]
        print(len(logits1))

        logits1_a = logits1_a.cpu().detach().numpy()
        logits1_a = logits1_a.reshape(1, -1)
        logits1_a = (logits1_a.tolist())[0]
        print(len(logits1_a))

        logits1_b = logits1_b.cpu().detach().numpy()
        logits1_b = logits1_b.reshape(1, -1)
        logits1_b = (logits1_b.tolist())[0]
        print(len(logits1_b))



        df1 = df.loc[:, ['temp']]
        df1 = np.array(df1.stack())
        df1 = df1.tolist()
        del df1[:int(len(df1) * 0.8)]
        print(df1, len(df1), type(df1))

        df1_a = df_a.loc[:,['temp']]
        df1_a = np.array(df1_a.stack())
        df1_a = df1_a.tolist()

        df1_b = df_b.loc[:, ['temp']]
        df1_b = np.array(df1_b.stack())
        df1_b = df1_b.tolist()

        # del df1[:int(len(df1) * 0.8)]
        # del df1_a[:int(len(df1_a))]
        # del df1_b[:int(len(df1_b))]
        print(df1_b, len(df1_b), type(df1_b))



        index = [i for i in range(len(df1))]
        index_a = [i for i in range(len(df1_a))]
        index_b = [i for i in range(len(df1_b))]

        df11 = pd.DataFrame(df1, columns = ['temp'])
        logits11 = pd.DataFrame(logits1, columns = ['prediction'])

        df11_a = pd.DataFrame(df1_a, columns=['temp'])
        logits11_a = pd.DataFrame(logits1_a, columns=['prediction'])

        df11_b = pd.DataFrame(df1_b, columns=['temp'])
        logits11_b = pd.DataFrame(logits1_b, columns=['prediction'])


        df11['time'] = index
        logits11['time'] = index

        df11_a['time'] = index_a
        logits11_a['time'] = index_a

        df11_b['time'] = index_b
        logits11_b['time'] = index_b

        print(df11, logits11, type(df11), type(logits11), end = '\n')

        # 创建第一个图形对象
        fig, ax = plt.subplots()
        fig_ab, ax_ab = plt.subplots()

        # 绘制第一个数据集
        ax.plot(df11['time'], df11['temp'], label='actual value')
        ax_ab.plot(df11_a['time'], df11_a['temp'], label='AAOI actual value')
        ax_ab.plot(df11_b['time'], df11_b['temp'], label='AAT actual value')

        # 绘制第二个数据集
        ax.plot(logits11['time'], logits11['prediction'], label='predicted value')
        ax_ab.plot(logits11_a['time'], logits11_a['prediction'], label='AAOI predicted value')
        ax_ab.plot(logits11_b['time'], logits11_b['prediction'], label='AAT predicted value')

        # 添加图例
        ax.legend()
        ax_ab.legend()

        # 添加标题和标签
        ax.set_title('test set from ACET')
        ax.set_xlabel('time')
        ax.set_ylabel('actual/predicted value')

        ax_ab.set_title('AAOI/AAT')
        ax_ab.set_xlabel('time')
        ax.set_ylabel('actual/predicted value')

        # 显示图形
        plt.show()

# 画 连接率-运行时间 和 连接率-test loss 图像
with open('df_connectivity_loss.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for row in dic.items():
        writer.writerow(row)

with open('df_connectivity_time.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for row in dic1.items():
        writer.writerow(row)

df_connectivity_loss = pd.read_csv('df_connectivity.csv')
df_connectivity_time = pd.read_csv('df_connectivity_time.csv')
print(df_connectivity_loss)

df_connectivity_loss.columns = ['connectivity ratio', 'test loss']
df_connectivity_time.columns = ['connectivity ratio', 'time']

# df_connectivity_loss.plot(x = 'connectivity ratio', y = 'test loss', kind='line')
# df_connectivity_time.plot(x = 'connectivity ratio', y = 'time', kind='line')
# plt.show()

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('connectivity ratio(%)')
ax1.set_ylabel('test loss(%)', color=color)
ax1.plot(df_connectivity_loss['connectivity ratio'], df_connectivity_loss['test loss'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # 创建共用x轴的第二个y轴

color = 'tab:blue'
ax2.set_ylabel('time(s)', color=color)
ax2.plot(df_connectivity_time['connectivity ratio'], df_connectivity_time['time'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()



# 画 AAOI 和 AAT 连接率-test loss 图像
with open('df_connectivity_loss_a.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for row in d1.items():
        writer.writerow(row)

with open('df_connectivity_loss_b.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for row in d2.items():
        writer.writerow(row)

df_connectivity_loss_a = pd.read_csv('df_connectivity_loss_a.csv')
df_connectivity_loss_b = pd.read_csv('df_connectivity_loss_b.csv')


df_connectivity_loss_a.columns = ['connectivity ratio', 'test_loss_a']
df_connectivity_loss_b.columns = ['connectivity ratio', 'test_loss_b']

# df_connectivity_loss.plot(x = 'connectivity ratio', y = 'test loss', kind='line')
# df_connectivity_time.plot(x = 'connectivity ratio', y = 'time', kind='line')
# plt.show()

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('connectivity ratio(%)')
ax1.set_ylabel('test loss_a(%)', color=color)
ax1.plot(df_connectivity_loss_a['connectivity ratio'], df_connectivity_loss_a['test_loss_a'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # 创建共用x轴的第二个y轴

color = 'tab:blue'
ax2.set_ylabel('test_loss_b(%)', color=color)
ax2.plot(df_connectivity_loss_b['connectivity ratio'], df_connectivity_loss_b['test_loss_b'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()
