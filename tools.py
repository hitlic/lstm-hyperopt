# coding: utf-8
"""
@author Liuchen
2018
"""
import numpy as np
from collections import Counter
try:
    from pyhanlp import HanLP as hanlp
except Exception:
    pass
import logging
logger = logging.getLogger('main.data_tools')


def load_embedding(embedding_file):
    """
    加载词向量，返回词典和词向量矩阵
    :param embedding_file: 词向量文件
    :return: tuple, (词典, 词向量矩阵)
    """
    logger.info('loading word dict and word embedding...')
    with open(embedding_file, encoding='utf-8') as f:
        lines = f.readlines()
        embedding_tuple = [tuple(line.strip().split(' ', 1)) for line in lines]
        embedding_tuple = [(t[0].strip().lower(), list(map(float, t[1].split()))) for t in embedding_tuple]
    embedding_matrix = []
    embedding_dim = len(embedding_tuple[0][1])
    embedding_matrix.append([0] * embedding_dim)  # 首行全为0，表示未登录词
    word_dict = dict()
    word_dict[''] = 0  # 空字符串表示未登录词
    word_id = 1
    for word, embedding in embedding_tuple:
        if word_dict.get(word) is None:
            word_dict[word] = word_id
            word_id += 1
            embedding_matrix.append(embedding)
    return word_dict, np.asarray(embedding_matrix, dtype=np.float32)


def drop_empty_texts(texts, labels):
    """
    去除预处理后句子为空的评论
    :param texts: id形式的文本列表
    :param labels: 标记数据
    :return: tuple of arrays. 非空句子列表，非空标记列表
    """
    logger.info("clear empty sentences ...")
    non_zero_idx = [id_ for id_, text in enumerate(texts) if len(text) != 0]
    texts_non_zero = np.array([texts[id_] for id_ in non_zero_idx])
    labels_non_zero = np.array([labels[id_] for id_ in non_zero_idx])
    return texts_non_zero, labels_non_zero


def make_dictionary_by_text(words_list):
    """
    构建词典（不使用已训练词向量时构建词典）
    :param words: list; 全部数数的词序列
    :return: tuple; 两个词典，word to int， int to word
    """
    logger.info("make dictionary by text ...")
    word_counts = Counter(words_list)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    id_to_word = {id_: word for id_, word in enumerate(sorted_vocab, 1)}
    word_to_id = {word: id_ for id_, word in id_to_word.items()}
    word_to_id[''] = 0
    id_to_word[0] = ''
    return word_to_id, id_to_word


def segment(text):
    '''
    使用HanLP对中文句子进行分词
    '''
    try:
        seg_result = hanlp.segment(text)
        return [term.word for term in seg_result]
    except Exception:
        return text.split()


def sentences2wordlists(sentence_list, lang='EN'):
    """
    将句子切分成词列表
    :param sentence_list: 句子列表
    :return: 词列表的列表
    """
    logger.info("word cutting ...")
    word_list_s = []
    for sentence in sentence_list:
        if lang == 'EN':  # 英文分词
            word_list = sentence.split()
        else:  # 中文分词
            word_list = segment(sentence)
        word_list_s.append(word_list)
    return word_list_s


def wordlists2idlists(word_list_s, word_to_id):
    """
    句子列表转id列表的列表
    :param word_list_s: 词列表的列表
    :param word_to_id: 词典
    :return: list of ints. id形式的句子列表
    """
    logger.info("convert word list to id list ...")
    sent_id_list = []
    for word_list in word_list_s:
        sent_id_list.append([word_to_id.get(word, 0) for word in word_list])
    return np.array(sent_id_list)


def labels2onehot(labels, class_num=None, class_labels=None):
    """
    生成句子的情感标记。调用时class_num与class_labels必选其一。
    :param labels: list; 数据的标记列表
    :param class_num: int; 类别总数
    :param class_labels: list; 类别标记，如[0, 1]、['a', 'b']
    :return: numpy array.
    """
    if class_num is None and class_labels is None:
        raise Exception("Parameter eithor class_num or class_labels must be given!  -- by lic")
    if class_labels is not None:
        class_num = len(class_labels)

    def label2onehot(label_):
        if class_labels is None:
            label_index = label_
        else:
            label_index = class_labels.index(label_)
        onehot_label = [0] * class_num
        onehot_label[label_index] = 1
        return onehot_label

    return np.array([label2onehot(label_) for label_ in labels])


def dataset_padding(text_ids, sent_len):
    """
    句子id列表左侧补0
    :param text_ids: id形式的句子列表
    :param seq_ken:  int, 最大句长
    :return: numpy array.  补0后的句子
    """
    logger.info("padding dataset ...")
    textids_padded = np.zeros((len(text_ids), sent_len), dtype=int)
    for i, row in enumerate(text_ids):
        textids_padded[i, -len(row):] = np.array(row)[:sent_len]

    return np.array(textids_padded)


def dataset_split(texts, labels, train_percent, random_seed=None):
    """
    训练、开发、测试集划分，其中训练集比例为train_percent，开发集和测试各集为0.5(1-train_percent)
    :param text: 数据集x
    :param labels: 数据集标记
    :param train_percent: 训练集所占比例
    :return: (train_x, train_y, val_x, val_y, test_x, test_y)
    """
    logger.info("split dataset ...")
    # 检测x与y长度是否相等
    assert len(texts) == len(labels)
    # 随机化数据
    if random_seed:
        np.random.seed(random_seed)
    shuf_idx = np.random.permutation(len(texts))
    texts_shuf = np.array(texts)[shuf_idx]
    labels_shuf = np.array(labels)[shuf_idx]

    # 切分数据
    split_idx = int(len(texts_shuf)*train_percent)
    train_x, val_x = texts_shuf[:split_idx], texts_shuf[split_idx:]
    train_y, val_y = labels_shuf[:split_idx], labels_shuf[split_idx:]

    test_idx = int(len(val_x)*0.5)
    val_x, test_x = val_x[:test_idx], val_x[test_idx:]
    val_y, test_y = val_y[:test_idx], val_y[test_idx:]

    return train_x, train_y, val_x, val_y, test_x, test_y


def make_batches(x, y, batch_size=100, shuffle=True):
    """
    将数据划分成训练批次
    :param x: 训练数据
    :param y: 训练数所标记
    :param batch_size: int, 批次大小
    :return: x和y的批次数据生成器
    """
    batch_size = int(batch_size)
    if shuffle:
        shuf_idx = np.random.permutation(len(x))
        x = np.array(x)[shuf_idx]
        y = np.array(y)[shuf_idx]
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for id_ in range(0, len(x), batch_size):
        yield x[id_:id_+batch_size], y[id_:id_+batch_size]


import shutil
import os


def rm_dirs(path, f_name, include=True):
    """
    当include=True时，删除path文件夹中名称中包含f_name的文件和文件夹
    当include=False时，删除path文件夹中名称中不包含f_name的文件和文件夹
    """
    files = os.listdir(path)
    for file in files:
        if include:
            if f_name in file:
                shutil.rmtree(os.path.join(path, file))
        else:
            if f_name not in file:
                shutil.rmtree(os.path.join(path, file))


class Parameters:
    '''
    超参数对像，用于存储与管理各种参数
    '''

    def __init__(self, **args):
        self.__dict__.update(args)  # 将参数加入到self中

    def __add__(self, hps):
        '''
        重载+，使两个超参对像可以相加
        '''
        if not isinstance(hps, Parameters):
            raise Exception(f'{type(self)} and {type(hps)} cannot be added together！！！ --- by LIC ')
        param_dict = dict()
        param_dict.update(self.__dict__)
        param_dict.update(hps.__dict__)
        return Parameters(** param_dict)

    def to_str(self, short=False):
        '''
        输出参数为字符串
        '''
        params = sorted(self.__dict__.items(), key=lambda item: item[0])
        output = ''
        for param, value in params:
            if short:
                output += f'{param}-{str(value)[:7]}__'
            else:
                output += f'{param}-{value}__'
        return output[:-2]

    def __str__(self):
        return self.to_str()

    def to_dict(self):
        '''
        将全部超参数输出为字典
        '''
        return self.__dict__

    def get(self, attr_name):
        '''
        获取参数值，若不存在返回None
        '''
        return self.__dict__.get(attr_name)

    def set(self, key, value):
        '''
        添加或更新一个参数
        '''
        self.__dict__[key] = value

    def default(self, default_params):
        '''
        设置默认参数，仅添加缺失的参数，不改变已有参数
        '''
        for key, value in default_params.items():
            if not self.__dict__.get(key):
                self.__dict__[key] = value

    def update(self, params):
        '''
        仅更新已有参数，不添加新参数
        '''
        param_set = params
        if isinstance(params, Parameters):
            param_set = params.to_dict()
        for key, value in param_set.items():
            if self.__dict__.get(key):
                self.__dict__[key] = value

    def extend(self, params):
        '''
        添加新参数，同时更新已有参数
        '''
        self.__dict__.update(params)


if __name__ == "__main__":
    rm_dirs('.', 'test')
