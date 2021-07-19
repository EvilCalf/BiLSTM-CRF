import os

from tensorflow.keras.preprocessing.sequence import pad_sequences

from BiLSTMCRF import BiLSTMCRF

vocab_path = 'model/vocab.txt'
train_path = 'train/sict_train.txt'
model_path = 'model/model.h5'
class_dict = {
            'O': 0,  # 非实体
            'B-DISEASE': 1,  # 疾病
            'I-DISEASE': 2,
            'B-SIGN': 3,  # 异常检查结果
            'I-SIGN': 4,
            'B-MARGIN': 5,  # 边缘
            'I-MARGIN': 6,
            'B-DIAMETER': 7,  # 直径
            'I-DIAMETER': 8,
            'B-TESTPROC': 9,  # 检查过程
            'I-TESTPROC': 10,
            'B-TREATMENT': 11,  # 治疗过程
            'I-TREATMENT': 12,
            'B-ORGAN': 13,  # 器官
            'I-ORGAN': 14,
            'B-ANATOMY': 15,  # 部位
            'I-ANATOMY': 16,
            'B-NATURE': 17,  # 性质
            'I-NATURE': 18,
            'B-SHAPE': 19,  # 形状
            'I-SHAPE': 20,
            'B-DENSITY': 21,  # 密度
            'I-DENSITY': 22,
            'B-BOUNDARY': 23,  # 边界
            'I-BOUNDARY': 24,
            'B-LUNGFIELD': 25,  # 肺野
            'I-LUNGFIELD': 26,
            'B-TEXTURE': 27,  # 纹理
            'I-TEXTURE': 28,
            'B-TRANSPARENCY': 29,  # 透明度
            'I-TRANSPARENCY': 30,
            'B-QUANTITY': 31,  # 数量
            'I-QUANTITY': 32
        }


def build_data():
    datas = []
    sample_x = []
    sample_y = []
    vocabs = {'UNK'}
    for line in open(train_path, 'r', encoding='utf-8'):
        line = line.rstrip().split('\t')
        if not line:
            continue
        char = line[0]
        if not char:
            continue
        cate = line[-1]
        sample_x.append(char)
        sample_y.append(cate)
        vocabs.add(char)
        if char in ['。', '?', '!', '！', '？']:
            datas.append([sample_x, sample_y])
            sample_x = []
            sample_y = []
    if os.path.exists(vocab_path):
        word_dict = load_worddict()
        print('读取现存词表！')
    else:
        word_dict = {wd: index for index, wd in enumerate(list(vocabs))}
        write_file(list(vocabs), vocab_path)
        print('保存词表！')
    return datas, word_dict


def write_file(wordlist, filepath):
    with open(filepath, 'w+', encoding='utf-8') as f:
        f.write('\n'.join(wordlist))


def modify_data(datas, word_dict, maxLen):
    x_train = [[word_dict[char] for char in data[0]]
               for data in datas]
    y_train = [[class_dict[label] for label in data[1]]
               for data in datas]
    x_train = pad_sequences(x_train, padding='post', maxlen=maxLen)
    y_train = pad_sequences(y_train, padding='post', maxlen=maxLen)
    return x_train, y_train


def load_worddict():
    vocabs = [line.strip()
              for line in open(vocab_path, encoding='utf-8')]
    word_dict = {wd: index for index, wd in enumerate(vocabs)}
    return word_dict


datas, word_dict = build_data()
vocabSize = len(word_dict)+1
maxLen = max(len(row[0]) for row in datas)
sequenceLengths = [len(row[0]) for row in datas]
x_train, y_train = modify_data(datas, word_dict, maxLen)
classSum = len(class_dict)
with open('config.txt', 'w+', encoding='utf-8') as f:
    f.write('vocabSize:'+str(vocabSize)+'\n')
    f.write('maxLen:'+str(maxLen)+'\n')
    f.write('classSum:'+str(classSum)+'\n')
f.close()

model = BiLSTMCRF(vocabSize, maxLen, class_dict, classSum, sequenceLengths)
model.net.summary()
if os.path.exists(model_path):
    model.load_weights(model_path)
    print('加载现有模型权重！')

history = model.fit(x_train, y_train, epochs=99999, batchsize=64)
