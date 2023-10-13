import pickle
from gensim.models import word2vec  # 引入word2vec
import re   #正则表达库

# 读入清洗后的数据
with open("train_set.pkl", "rb") as f:
    train_set = pickle.load(f)
with open("validate_set.pkl", "rb") as f:
    validate_set = pickle.load(f)
with open("test_set.pkl", "rb") as f:
    test_set = pickle.load(f)


# sentence="One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. They are right, as this is exactly what happened with me.<br /><br />The first thing that struck me about Oz was its brutality and unflinching scenes of violence, which set in right from the word GO. Trust me, this is not a show for the faint hearted or timid. This show pulls no punches with regards to drugs, sex or violence. Its is hardcore, in the classic use of the word.<br /><br />It is called OZ as that is the nickname given to the Oswald Maximum Security State Penitentary. It focuses mainly on Emerald City, an experimental section of the prison where all the cells have glass fronts and face inwards, so privacy is not high on the agenda. Em City is home to many..Aryans, Muslims, gangstas, Latinos, Christians, Italians, Irish and more....so scuffles, death stares, dodgy dealings and shady agreements are never far away.<br /><br />I would say the main appeal of the show is due to the fact that it goes where other shows wouldn't dare. Forget pretty pictures painted for mainstream audiences, forget charm, forget romance...OZ doesn't mess around. The first episode I ever saw struck me as so nasty it was surreal, I couldn't say I was ready for it, but as I watched more, I developed a taste for Oz, and got accustomed to the high levels of graphic violence. Not just violence, but injustice (crooked guards who'll be sold out for a nickel, inmates who'll kill on order and get away with it, well mannered, middle class inmates being turned into prison bitches due to their lack of street skills or prison experience) Watching Oz, you may become comfortable with what is uncomfortable viewing....thats if you can get in touch with your darker side."
# fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
#             '\?', '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”',
#             '“', ]
# sentence = sentence.lower()  # 把大写转化为小写
# sentence = re.sub("<br />", " ", sentence)
# sentence = re.sub("|".join(fileters), " ", sentence)  # 正则表达，代换filters的符号
# sentence = [i for i in sentence.split(" ") if len(i) > 0]

def remove_stopwords(sentence):
    stopwords = []
    with open("stopword.txt") as f:
        lines = f.readlines()
    for line in lines:
        line = line.rstrip("\n")
        stopwords.append(line)

    for sw in stopwords:
        while sw in sentence:
            sentence.remove(sw)
    return sentence

# sentence = remove_stopwords(sentence)
# print(sentence)
#print(stopwords) #["'d", "'ll", "'m", "'re", "'s", "'t", "'ve", 'ZT',...]


for sentence in train_set:
    sentence = remove_stopwords(sentence)
with open("train_set_remove_sw.pkl", "wb") as f:
    pickle.dump(train_set, f)

for sentence in validate_set:
    sentence = remove_stopwords(sentence)
with open("validate_set_remove_sw.pkl", "wb") as f:
    pickle.dump(validate_set, f)

for sentence in test_set:
    sentence = remove_stopwords(sentence)
with open("test_set_remove_sw.pkl", "wb") as f:
    pickle.dump(test_set, f)
