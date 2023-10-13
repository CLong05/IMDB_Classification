import pickle
from gensim.models import word2vec  # 引入word2vec

# 读入清洗后的数据
with open("train_set_remove_sw.pkl", "rb") as f:
    train_set = pickle.load(f)
with open("validate_set_remove_sw.pkl", "rb") as f:
    validate_set = pickle.load(f)
with open("test_set_remove_sw.pkl", "rb") as f:
    test_set = pickle.load(f)

# 汇总词汇
sentences = train_set + validate_set + test_set

# 构建模型
model = word2vec.Word2Vec(sentences, min_count=100, vector_size=10)
# 保存模型
model.save('model')
# 加载模型
model = word2vec.Word2Vec.load('model')

# 存储词向量
model.save("./IMDB_word2vec.model")
model.wv.save_word2vec_format("./IMDB_wor2vec.txt")

