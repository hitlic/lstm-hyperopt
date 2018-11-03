# coding: utf-8
"""
@author Liuchen
2018
"""
import tensorflow as tf
import tools
import numpy as np
import os
import time
import logging
logger = logging.getLogger('main.dnn_model')


class DNNModel:
    def __init__(self, hyper_params=None):
        '''
        超参数输入方法1：以HyperParams对象的方式输入参数

        参数：
        class_num       分类类别数量
        embed_size      词向量维度
        lstm_sizes      RNN隐层维度，可有多层RNN
        vocab_size      词典大小
        embed_matrix    词向量矩阵
        fc_size         全连接层大小
        max_sent_len    最大句长
        isBiRNN         * 是否使用双向RNN
        refine          * 词向量是否refine
        '''
        tf.reset_default_graph()  # 清空计算图，否则连续多次执行会出错
        if hyper_params and not isinstance(hyper_params, tools.Parameters):
            raise Exception(f'hyper_params must be an object of {type(tools.Parameters)} --- by LIC')

        # 默认参数
        default_params = {
            'isBiRNN': True,
            'refine': False,
        }
        hyper_params.default(default_params)
        self.hypers = hyper_params  # 所有超参数都保存在其中

    def set_placeholers(self):
        '''
        placeholder 参数
        '''
        self.learning_rate = tf.placeholder_with_default(0.01, shape=(), name='learning_rate')  # 学习率
        self.keep_prob = tf.placeholder_with_default(
            1.0, shape=(), name='keep_prob')                                 # dropout keep probability
        self.l2reg = tf.placeholder_with_default(0.0, shape=(), name='L2reg')              # L2正则化参数

    def inputs_layer(self):
        """
        输入层
        """
        with tf.name_scope('input_layer'):
            # 数据x placeholder
            self.inputs = tf.placeholder(tf.int32, [None, self.hypers.max_sent_len], name='inputs')
            # 标记y placeholder
            self.labels = tf.placeholder(tf.int32, [None, self.hypers.class_num], name='labels')
        return self.inputs

    def embedding_layer(self, inputs_):
        """
        词向量层
        """
        with tf.name_scope("embedding_layer"):
            if self.hypers.get('embed_matrix') is None:   # 若无已训练词向量
                embedding = tf.Variable(tf.random_uniform((self.hypers.vocab_size,
                                                           self.hypers.embed_dim), -1, 1), name="embedding")
            else:                           # 若已有词向量
                embedding = tf.Variable(self.hypers.embed_matrix,
                                        trainable=self.hypers.refine, name="embedding")
            embed = tf.nn.embedding_lookup(embedding, inputs_)
        return embed

    def rnn_layer(self, embed):
        """
        RNN层
        """
        with tf.name_scope("rnn_layer"):
            embed = tf.nn.dropout(embed, keep_prob=self.keep_prob)  # dropout
            # --- 可选的RNN单元
            # tf.contrib.rnn.BasicRNNCell(size)
            # tf.contrib.rnn.BasicLSTMCell(size)
            # tf.contrib.rnn.LSTMCell(size)
            # tf.contrib.rnn.GRUCell(size, activation=tf.nn.relu)
            # tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(size)
            # tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(size)

            if not self.hypers.isBiRNN:
                lstms = [tf.contrib.rnn.BasicLSTMCell(size) for size in self.hypers.rnn_dims]
                drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.keep_prob) for lstm in lstms]
                cell = tf.contrib.rnn.MultiRNNCell(drops)  # 组合多个 LSTM 层
                lstm_outputs, _ = tf.nn.dynamic_rnn(cell, embed, dtype=tf.float32)
                # lstm_outputs -> batch_size * max_len * n_hidden
            else:
                lstms_l = [tf.contrib.rnn.BasicLSTMCell(size) for size in self.hypers.rnn_dims]
                lstms_r = [tf.contrib.rnn.BasicLSTMCell(size) for size in self.hypers.rnn_dims]
                drops_l = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.keep_prob) for lstm in lstms_l]
                drops_r = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.keep_prob) for lstm in lstms_r]
                cell_l = tf.contrib.rnn.MultiRNNCell(drops_l)
                cell_r = tf.contrib.rnn.MultiRNNCell(drops_r)
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(  # 双向LSTM
                    cell_l,  # 正向LSTM单元
                    cell_r,  # 反向LSTM单元
                    inputs=embed,
                    dtype=tf.float32,
                )  # outputs -> batch_size * max_len * n_hidden; state(最终状态，为h和c的tuple) -> batch_size * n_hidden
                lstm_outputs = tf.concat(outputs, -1)  # 合并双向LSTM的结果
            outputs = lstm_outputs[:, -1]  # 返回每条数据的最后输出

        return outputs

    def fc_layer(self, inputs):
        """
        全连接层
        """
        # initializer = tf.contrib.layers.xavier_initializer()  # xavier参数初始化，暂没用到
        with tf.name_scope("fc_layer"):
            inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob, name='drop_out')  # dropout
            # outputs = tf.contrib.layers.fully_connected(inputs, self.hypers.fc_size, activation_fn=tf.nn.relu)
            outputs = tf.layers.dense(inputs, self.hypers.fc_size, activation=tf.nn.relu)
        return outputs

    def output_layer(self, inputs):
        """
        输出层
        """
        with tf.name_scope("output_layer"):
            inputs = tf.layers.dropout(inputs, rate=1-self.keep_prob)
            outputs = tf.layers.dense(inputs, self.hypers.class_num, activation=None)
            # outputs = tf.contrib.layers.fully_connected(inputs, self.hypers.class_num, activation_fn=None)
        return outputs

    def set_loss(self):
        """
        损失函数
        """
        # softmax交叉熵损失
        with tf.name_scope("loss_scope"):
            reg_loss = tf.contrib.layers.apply_regularization(  # L2正则化
                tf.contrib.layers.l2_regularizer(self.l2reg),
                tf.trainable_variables()
            )
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.predictions, labels=self.labels)) + reg_loss   # ---GLOBAL---损失函数

    def set_accuracy(self):
        """
        准确率
        """
        with tf.name_scope("accuracy_scope"):
            correct_pred = tf.equal(tf.argmax(self.predictions, axis=1), tf.argmax(self.labels, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))   # ---GLOBAL---准确率

    def set_optimizer(self):
        """
        优化器
        """
        with tf.name_scope("optimizer"):
            # --- 可选优化算法
            # self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(self.loss)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            # self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)
            # self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.loss)
            # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
            # self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

    def build(self):
        """
        DNN模型构建
        """
        self.set_placeholers()

        inputs = self.inputs_layer()
        embedding = self.embedding_layer(inputs)
        rnn_outputs = self.rnn_layer(embedding)
        fc_outputs = self.fc_layer(rnn_outputs)
        self.predictions = self.output_layer(fc_outputs)
        self.set_loss()
        self.set_optimizer()
        self.set_accuracy()


def train(dnn_model, params):
    """
    训练并验证
    :param dnn_model: 神经网络模型
    :param params: 训练参数
    :return: 最小验证损失,最大验证准确率,训练迭代次数
    params.dnn_model         计算图模型
    params.learning_rate     学习率
    params.train_x           训练数据
    params.train_y           标记训练数据
    params.dev_x             验证数据
    params.dev_y             标记验证数据
    params.max_epochs        最大迭代次数
    params.batch_size        minibatch 大小
    params.keep_prob         dropout keep probability
    params.l2reg             L2 正则化系数
    params.show_step         * 隔多少步显示一次训练结果
    params.checkpoint_path   * 模型保存位置
    params.model_name        * 保存下来的模型的名称（文件夹名）
    params.no_improve        * early stop, 连续no_improve次没有得到更低的dev_loss则停止
    """
    if params and not isinstance(params, tools.Parameters):
        raise Exception(f'Parameter params must be an object of {type(tools.Parameters)} -- by LIC')

    default_params = {
        'show_step': 0,
        'checkpoint_path': "./checkpoints",
        'model_name': None,
        'no_improve': 5
    }
    params.default(default_params)

    # 最佳模型保存路径
    if params.get('model_name') is None:
        params.set('model_name', str(time.time()).replace('.', '')[:11])
    best_model_path = params.checkpoint_path + '/best/' + params.model_name
    if not os.path.exists(best_model_path):  # 模型保存路径不存在则创建路径
        os.makedirs(best_model_path)
    saver = tf.train.Saver()

    with tf.Session() as sess:

        # Train Summaries
        train_loss = tf.summary.scalar("train_loss", dnn_model.loss)
        train_acc = tf.summary.scalar("train_acc", dnn_model.accuracy)
        train_summary_op = tf.summary.merge([train_loss, train_acc])
        train_summary_writer = tf.summary.FileWriter('./log/train', sess.graph)

        # Dev summary writer
        dev_summary_writer = tf.summary.FileWriter('./log/dev', sess.graph)

        # meta日志
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)  # for meta日志  **1
        # run_metadata = tf.RunMetadata()                                    # for meta日志  **2

        sess.run(tf.global_variables_initializer())
        n_batches = len(params.train_x)//int(params.batch_size)
        step = 0
        best_dev_acc = 0     # 最优验证准确率
        min_dev_loss = float('inf')  # 最小验证损失

        no_improve_num = 0    # 最优dev loss 没有下降的次数
        epoch_num = 0
        for e in range(params.max_epochs):
            epoch_num = e + 1
            for id_, (x, y) in enumerate(tools.make_batches(params.train_x, params.train_y, int(params.batch_size)), 1):
                step += 1
                feed = {
                    dnn_model.inputs: x,
                    dnn_model.labels: y,
                    dnn_model.learning_rate: params.learning_rate,
                    dnn_model.keep_prob: params.keep_prob,
                    dnn_model.l2reg: params.l2reg
                }
                train_loss, _, train_acc, train_summary = sess.run(
                    [dnn_model.loss, dnn_model.optimizer, dnn_model.accuracy, train_summary_op],
                    feed_dict=feed,
                    # options=run_options,                                  # for meta 日志 - **3
                    # run_metadata=run_metadata                             # for meta 日志 - **4
                )

                train_summary_writer.add_summary(train_summary, step)  # 写入日志
                # --- 写入meta日志，注意：日志文件会特别巨大，若要写入meta日志需取消 **1行、**2行、**3行、**4行和**5行的注释
                # train_summary_writer.add_run_metadata(run_metadata, 'batch%03d' % step) # for meta 日志 - **5

                if params.show_step > 0 and step % params.show_step == 0:
                    info = "Epoch {}/{} ".format(e+1, params.max_epochs) + \
                        " - Batch {}/{} ".format(id_+1, n_batches) + \
                        " - Loss {:.5f} ".format(train_loss) + \
                        " - Acc {:.5f}".format(train_acc)
                    logger.info(info)

            # 每个 Epoch 验证 ---
            dev_acc_s = []
            dev_loss_s = []
            for xx, yy in tools.make_batches(params.dev_x, params.dev_y, params.batch_size):
                feed = {
                    dnn_model.inputs: xx,
                    dnn_model.labels: yy,
                    dnn_model.keep_prob: 1,
                }
                dev_batch_loss, dev_batch_acc = sess.run([dnn_model.loss, dnn_model.accuracy], feed_dict=feed)
                dev_acc_s.append(dev_batch_acc)
                dev_loss_s.append(dev_batch_loss)

            dev_acc = np.mean(dev_acc_s)    # dev acc 均值
            dev_loss = np.mean(dev_loss_s)  # dev loss 均值

            # --- dev 日志
            dev_summary = tf.Summary()
            dev_summary.value.add(tag="dev_loss", simple_value=dev_loss)
            dev_summary.value.add(tag="dev_acc", simple_value=dev_acc)
            dev_summary_writer.add_summary(dev_summary, step)

            info = "|Epoch {}/{}\t".format(e+1, params.max_epochs) + \
                "|Train-Loss| {:.5f}\t".format(train_loss) + \
                "|Dev-Loss| {:.5f}\t".format(dev_loss) + \
                "|Train-Acc| {:.5f}\t".format(np.mean(train_acc)) + \
                "|Dev-Acc| {:.5f}".format(dev_acc)
            logger.info(info)


            # 保存最好的模型
            if best_dev_acc < dev_acc:
                best_dev_acc = dev_acc
                saver.save(sess, best_model_path + "/best_model.ckpt")
            # 保存每个epoch的模型
            # saver.save(sess, best_model_path + "model.ckpt", global_step=e)

            # 寻找最小 dev_loss
            if min_dev_loss > dev_loss:
                min_dev_loss = dev_loss
                no_improve_num = 0
            else:
                no_improve_num += 1

            # early stop
            if no_improve_num == params.no_improve:
                break

        logger.info("** The best dev accuracy: {:.5f}".format(best_dev_acc))

    # 返回最小的验证损失、最优准确率、迭代次数。注意：最优准确率并非意味着最小损失！
    return min_dev_loss, best_dev_acc, epoch_num


def test(dnn_model, test_x, test_y, batch_size=256, model_dir="./checkpoints/best", model_name=None):
    """
    利用最好的模型进行测试
    :param test_x: 测试数据
    :param test_y: 标记测试数据
    :param batch_size: 批次大小
    :param dnn_model: 原dnn模型
    :param model_dir: 训练好的模型的存储位置
    """
    # 若无model_name则利用最新模型，否则利用提供的模型
    if model_name is None:
        best_folder = max([d for d in os.listdir(model_dir) if d.isdigit()])
        best_model_dir = model_dir + '/' + best_folder
    else:
        best_model_dir = model_dir + '/' + model_name
    saver = tf.train.Saver()
    test_acc = []
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(best_model_dir))
        for _, (x, y) in enumerate(tools.make_batches(test_x, test_y, batch_size), 1):
            feed = {
                dnn_model.inputs: x,
                dnn_model.labels: y,
                dnn_model.keep_prob: 1,
            }
            batch_acc = sess.run([dnn_model.accuracy], feed_dict=feed)
            test_acc.append(batch_acc)
            acc_result = np.mean(test_acc)
        logger.info("** Test Accuracy: {:.5f}".format(acc_result))
    return acc_result
