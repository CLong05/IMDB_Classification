# IMDB电影评论文本分类  IMDB Film Review Text Classification

**简介**：独立搭建神经网络模型，实现根据情感态度对IMDB影评进行文本分类。

**具体内容**：不调用现有的pytorch等机器学习库，使用Python语言独立搭建前馈神经网络模型。对给定的影评信息进行处理，根据其TF、TF-IDF、word2vec特征分别训练出一个二分类神经网络，使得训练得到的3个神经网络模型能够判断输入的用户影评信息属于正面评价还是负面评价。

**文件说明**：

1.数据集预处理文件夹：进行数据预处理

2~4.剩余三个文件夹内容相似：

- 以tf前馈神经网络为例：
  - save_set_tf.pkl：保存了最后训练得到的神经网络模型，可在net_checkpoint.py中恢复并测试。
  - test_label_tf.pkl、test_set_tf.pkl：前馈神经网络所需的测试集数据。
  - 训练过程以及测试集表现.txt：保存了模型训练、模型测试过程中中准确率等信息
  - 各个.py文件都在报告对应的部分中有做说明，因此不再赘述。      

------

**Introduction**: A neural network model was built independently to classify IMDB film reviews according to emotional attitudes.

**Specific content**: Without calling existing machine learning libraries such as Pytorch, Python language is used to independently build feed-forward neural network models. After processing the given movie review information, a binary classification neural network is trained according to its TF, TF-IDF and word2vec characteristics respectively, so that the three trained neural network models can judge whether the input user movie review information belongs to positive evaluation or negative evaluation.

**Document description:**

1.Data set preprocessing folder: Perform data preprocessing

2 to 4. The contents of the remaining three folders are similar:

- Take tf feedforward neural network as an example:
  - save_set_tf.pkl: saves the last trained neural network model, which can be recovered and tested in net_checkpoint.py.
  - test_label_tf.pkl, test_set_tf.pkl: Test set data required by the feedforward neural network.
  - 训练过程以及测试集表现.txt: saves information such as accuracy during model training and model testing
  - Each <u>.py</u> file is described in the corresponding section of the report, so it will not be described here.