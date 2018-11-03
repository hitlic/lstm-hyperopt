# coding: utf-8
"""
@author Liuchen
2018
"""

import tools
from tools import Parameters
import numpy as np
import dnn_model as dm
import data_prepare as dp
from itertools import chain
from hyperopt import fmin, hp, Trials, tpe, STATUS_OK
import time
import pprint
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
logger = logging.getLogger('main')
pp = pprint.PrettyPrinter(indent=1,width=150,depth=None)

# ================== step0: 参数准备 =================

tune_params = Parameters(**{
    'learning_rate': 0.001,         # 学习率
    'batch_size': 64,               # mini batch大小
    'keep_prob': 0.5,               # drop out
    'l2reg': 0.001,                 # L2正则化
    'rnn_dims': [256],              # RNN隐层维度，可有多层RNN
})

# 网络结构参数
net_params = Parameters(**{
    'class_num': 2,                 # 分类类别数量
    'embed_dim': 300,               # 词向量维度
    'rnn_dims': [256],              # RNN隐层维度，可有多层RNN
    'fc_size': 2000,                 # 全连接层大小
    'max_sent_len': 60,             # 最大句长
    'refine': False                 # 词向量矩阵是否参与训练
})
net_params.update(tune_params)

# 训练参数
train_params = Parameters(**{
    'learning_rate': 0.001,         # 学习率
    'batch_size': 64,               # mini batch大小
    'keep_prob': 0.5,               # drop out
    'l2reg': 0.001,                 # L2正则化
    'show_step': 0,                # 中间结果显示
    'max_epochs': 30,               # 最大迭代次数
})
train_params.update(tune_params)

# 数据准备参数
lang = 'EN'              # 文本语言 EN为英文，CN为中文
train_percent = 0.8      # 训练数据的比例
data_path = '../data/'   # 数据存放路径

# ================== step1: 数据准备 =================
## a. 从csv文件读取数据
texts, labels = dp.load_from_csv(data_path + "data.csv")
## b. 从以Tab符为分隔符的csv文件读取数据
# texts, labels = dp.load_from_csv(data_path + "cn_data.txt", delimiter='\t', lang=lang)
## c. 从情感类别文件读取数据
# texts, labels = dp.load_from_class_files([data_path + 'pos.txt', data_path + 'neg.txt'])

# --- 分词（英文按空格分，中文利用hanlp分词）
texts = tools.sentences2wordlists(texts, lang=lang)
logger.info('max sentence len: ' + str(max([len(text) for text in texts])))

# --- 构建词典
## a. 基于文本构建词典  -- 不使用预训练词向量
vocab_to_int, int_to_vocab = tools.make_dictionary_by_text(list(chain.from_iterable(texts)))
embedding_matrix = None  # 设词向量矩阵为None

## b. 基于词向量构建词典 -- 使用预训练词向量
# vocab_to_int, embedding_matrix = tools.load_embedding(data_path + "word_embedding_300_new.txt") # 英文词向量
# vocab_to_int, embedding_matrix = tools.load_embedding(data_path + "glove.6B.200d.txt")  # 英文词向量
# vocab_to_int, embedding_matrix = tools.load_embedding(data_path + "sgns.weibo.word.txt") # 中文词向量

net_params.set('embedding_matrix', embedding_matrix)  # 添加词向量矩阵参数
net_params.set('vocab_size', len(vocab_to_int))      # 添加词典大小参数

logger.info(f"dictionary length: {len(vocab_to_int)}")

texts = tools.wordlists2idlists(texts, vocab_to_int)                        # 将句子转成词典id列表
texts, labels = tools.drop_empty_texts(texts, labels)                       # 清除预处理后文本为空的数据
labels = tools.labels2onehot(labels, net_params.class_num)                             # 将类别标记转为one-hot形式
texts = tools.dataset_padding(texts, sent_len=net_params.max_sent_len)                 # 左侧补0

lowerst_dev_loss = float('inf') 
# ================== step2: 定义调参目标函数 =================
def dev_loss(args):
    '''
    调参优化目标函数，使用了hold out验证
    '''
    tune_args = Parameters(**args)
    if tune_args.get('batch_size'):
        tune_args.batch_size = int(tune_args.batch_size)
    net_params.update(tune_args)
    train_params.update(tune_args)
    logger.info(f'\n\n \t <<<<<< ** parameters: {tune_args.to_str()} >>>>>>\n')

    HOLD_OUT = 5  # hold out 次数
    min_dev_loss = []
    max_dev_acc = []
    epochs = []
    for _ in range(HOLD_OUT):
        time_str = str(time.time()).replace('.', '')[:11]
        train_params.set('model_name', f'{time_str}__{tune_args.to_str()}')  # 模型保存位置
        # 数据集划分
        train_x, train_y, val_x, val_y, _, _ = tools.dataset_split(texts, labels, train_percent=train_percent)
        train_params.extend({
            'train_x': train_x,                 # 训练数据
            'train_y': train_y,                 # 训练数据标记
            'dev_x': val_x,                     # 验证数据
            'dev_y': val_y,                     # 验证数据标记
        })

        model = dm.DNNModel(net_params)
        model.build()

        min_dev_loss_, max_dev_acc_, epoch = dm.train(model, train_params)
        min_dev_loss.append(min_dev_loss_)
        max_dev_acc.append(max_dev_acc_)
        epochs.append(epoch)

    global lowerst_dev_loss
    if lowerst_dev_loss > np.average(min_dev_loss):  # 仅保留最好的参数对应的模型，其余删除
        lowerst_dev_loss = np.average(min_dev_loss)
        tools.rm_dirs('./checkpoints/best', tune_args.to_str(), False)
    else:
        tools.rm_dirs('./checkpoints/best', tune_args.to_str(), True)

    logger.info(f'\n\n\t >>>>>> Acc   {str(max_dev_acc)}\t{np.average(max_dev_acc)}\
             \n\t >>>>>> Loss  {str(min_dev_loss)}\t{np.average(min_dev_loss)}\n\t >>>>>> Epoch {str(epochs)}\n')

    return {
        'status': STATUS_OK,
        'loss': np.average(min_dev_loss),
        'epochs': epochs,
        'metrics': {
            'accuracys': max_dev_acc,
            'accuracy': np.average(max_dev_acc),
            'losses': min_dev_loss
        }
    }


# ================== step3: 调参 =================

space = {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.01)),   # 学习率
    'batch_size': hp.quniform('batch_size', 32, 512, 8),                             # mini batch大小
    'keep_prob': hp.uniform('keep_prob', 0.3, 1.0),                                  # drop out
    'l2reg': hp.loguniform('l2reg', np.log(0.0001), np.log(0.01)),                   # L2正则化
}

ts = Trials()
from functools import partial
best = fmin(dev_loss, space, algo=partial(tpe.suggest, n_startup_jobs=15), max_evals=50, trials=ts)
print('TPE best: {}'.format(best))

for trial in ts.trials:
    pp.pprint({
        'parameters': trial['misc']['vals'],
        'results': trial['result']
    })

# # ================== step4: 测试 =================
# dm.test(model, test_x, test_y, model_name=)
