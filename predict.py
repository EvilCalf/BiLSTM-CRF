import csv
import json
import os

from tensorflow.keras.preprocessing.sequence import pad_sequences

from BiLSTMCRF import BiLSTMCRF

model_path = 'model/model.h5'
vocab_path = 'model/vocab.txt'
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
maxLen = 198
classSum = 29


def build_input(text):
    x = []
    for char in text:
        if char not in word_dict:
            char = 'UNK'
        x.append(word_dict.get(char))
    x = pad_sequences([x], padding='post', maxlen=maxLen)
    return x


def load_worddict():
    vocabs = [line.strip()
              for line in open(vocab_path, encoding='utf-8')]
    word_dict = {wd: index for index, wd in enumerate(vocabs)}
    return word_dict


def predict(text):
    y_pre = []
    str = build_input(text)
    raw = model.predict(str)[0]
    chars = [i for i in text]
    tags = [label_dict[i] for i in raw][:len(text)]
    res = list(zip(chars, tags))
    for i, tag in enumerate(tags):
        y_pre.append(tag)
    return res, y_pre


def output(cnt):
    output = []
    flag = 0
    start = []
    end = []
    tags = []
    for i, tag in enumerate(cnt):
        if tag == 'O':
            if flag == 1:
                end = i-1
                output.append([tags, start, end])
            flag = 0
            continue
        if tag.split("-")[0] == 'B':
            if flag == 1:
                end = i-1
            flag = 1
            start = i
            tags = tag.split("-")[1]
            continue
    return output


word_dict = load_worddict()
vocabSize = len(word_dict)+1
label_dict = {j: i for i, j in class_dict.items()}

model = BiLSTMCRF(vocabSize=vocabSize, maxLen=maxLen,
                  tagIndexDict=class_dict, tagSum=classSum)
model.load_weights(model_path)

res = []
ans_json = []
for root, dirs, files in os.walk("data_out"):
    for file in files:
        with open(root+"/"+file, "r", encoding='utf-8') as f:
            reader = csv.reader(f)
            result = list(reader)
            total = len(result)
            for i, strs in enumerate(result):
                ans, y_pre = predict(strs[0])
                res.append(ans)
                print(str(i)+"/"+str(total-1))
                y_pre = output(y_pre)
                filename = open("data_out_json/"+file.replace(".csv", "") +
                                "_"+str(i)+"_P.txt", 'w+', encoding='utf-8')
                for value in y_pre:
                    filename.write(str(value)+'\n')
                filename.close()
            with open("data_out_json/"+file.replace(".csv", "")+".json", 'w', encoding='utf-8') as file_obj:
                for i, strs in enumerate(res):
                    ans_json.append(
                        {'data': [{'chars': index[0], 'tags': index[1]} for index in res[i]]})
                json.dump(ans_json, file_obj, indent=4, ensure_ascii=False)
            filename = open("data_out_json/"+file.replace(".csv",
                                                          "")+"_BIO.txt", 'w+', encoding='utf-8')
            for i, strs in enumerate(res):
                for j, value in enumerate(strs):
                    filename.write(str(value[0])+'\t'+str(value[1])+'\n')
            filename.close()
