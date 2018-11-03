# lstm-hyperopt
 一个基本的多层lstm rnn模型，使用hyperopt进行调参，保存调参过程中的最优模型并用于分类。

### 运行环境
- python 3.6

### 软件包依赖
- tensorflow 1.6及以上
- hyperopt
- pandas
- pyhanlp (可选，用于中文分词)

### 词向量
- data文件夹中的英文词向量维度为300，但词汇量太小，仅做测试用。可从这里下载替换 http://nlp.stanford.edu/data/glove.6B.zip
- 中文词向量参见  https://github.com/Embedding/Chinese-Word-Vectors

### 功能说明
- 多层lstm rnn模型
- 能够保存训练过程中最佳的模型，用于测试。（也可保存每个epoch的模型，去掉dnn_model.py中train方法相应部分代码的注释即可）
- early stop
- 能够输出日志，包括计算图，以及loss、train_accuracy、dev_accuracy，可利用tensorboard查看。（也可输出元日志，去除中train方法相应部分代码的注释即可，但日志文件相当巨大，且会明显影响训练速度，建议在必要时再打开）

### 拓展
- train.py用于单次训练模型；
- tune.py用于调参；
- 若数据格式有变，修改data_prepare.py，读取新格式的数据；
- 若模型有变，修改dnn_model.py，修改或添加自己设计好的layer，并修build方法。
