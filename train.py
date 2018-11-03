# coding: utf-8
"""
@author Liuchen
2018
"""

import tools
from tools import Parameters
import dnn_model as dm
import data_prepare as dp
from itertools import chain
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
logger = logging.getLogger('main')

# ================== step0: 参数准备 =================

tune_params = Parameters(**{
    'learning_rate': 0.002,         # 学习率
    'batch_size': 128,              # mini batch大小
    'keep_prob': 0.3,               # drop out
    'l2reg': 0.0016,                # L2正则化
    'rnn_dims': [256],              # RNN隐层维度，可有多层RNN
})

# -----------------------------------
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

# 数据集划分
train_x, train_y, val_x, val_y, test_x, test_y = tools.dataset_split(texts, labels, train_percent=train_percent)
train_params.extend({
    'train_x': train_x,                 # 训练数据
    'train_y': train_y,                 # 训练数据标记
    'dev_x': val_x,                     # 验证数据
    'dev_y': val_y,                     # 验证数据标记
})
# ================== step2: 构建模型 =================
model = dm.DNNModel(net_params)
model.build()

# ================== step3: 训练 =================
min_dev_loss = dm.train(model, train_params)
logger.info(f' ** The minimum dev_loss is {min_dev_loss}')

# ================== step4: 测试 =================
dm.test(model, test_x, test_y)
