import csv
import json
import os

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

from BiLSTMCRF import BiLSTMCRF

model_path='model/model.h5'
vocab_path='model/vocab.txt'
test_path='test/input.txt'
class_dict = {
            "O": 0,
            "B-DISEASE": 1,
            "I-DISEASE": 2,
            "B-SIGN": 3,
            "I-SIGN": 4,
            "B-MARGIN": 5,
            "I-MARGIN": 6,
            "B-DIAMETER": 7,
            "I-DIAMETER": 8,
            "B-TESTPROC": 9,
            "I-TESTPROC": 10,
            "B-TREATMENT": 11,
            "I-TREATMENT": 12,
            "B-ANATOMY": 13,
            "I-ANATOMY": 14,
            "B-NATURE": 15,
            "I-NATURE": 16,
            "B-SHAPE": 17,
            "I-SHAPE": 18,
            "B-DENSITY": 19,
            "I-DENSITY": 20,
            "B-BOUNDARY": 21,
            "I-BOUNDARY": 22,
            "B-LUNGFIELD": 23,
            "I-LUNGFIELD": 24,
            "B-TEXTURE": 25,
            "I-TEXTURE": 26,
            "B-TRANSPARENCY": 27,
            "I-TRANSPARENCY": 28
        }
maxLen=198
classSum=29

def build_input(text):
    x = []
    for char in text:
        if char not in word_dict:
            char = 'UNK'
        x.append(word_dict.get(char))
    x = pad_sequences([x],padding='post', maxlen=maxLen)
    return x

def load_worddict():
    vocabs = [line.strip()
              for line in open(vocab_path, encoding='utf-8')]
    word_dict = {wd: index for index, wd in enumerate(vocabs)}
    return word_dict

def predict(text): 
    y_pre=[]
    str = build_input(text)
    raw =  model.predict(str)[0]
    chars = [i for i in text]
    tags = [label_dict[i] for i in raw][:len(text)]
    res = list(zip(chars, tags))
    for i,tag in enumerate(tags):
        y_pre.append(tag)
    return res,y_pre

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

def build_data(test_path):
    datas = []
    sample_x = []
    sample_y = []
    for line in open(test_path, 'r', encoding='utf-8'):
        line = line.rstrip().split('\t')
        if not line:
            continue
        char = line[0]
        if not char:
            continue
        cate = line[-1]
        sample_x.append(char)
        sample_y.append(cate)
        if char in ['。', '?', '!', '！', '？']:
            datas.append([sample_x, sample_y])
            sample_x = []
            sample_y = []
    return datas

def modify_data(datas):
    x_test = []
    txt = ""
    for data in datas:
        txt = txt.join(str(i) for i in data[0])
        x_test.append(txt)
        txt = ""
    return x_test

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

model = BiLSTMCRF(vocabSize=vocabSize, maxLen=maxLen, tagIndexDict=class_dict, tagSum=classSum)
model.load_weights(model_path)

datas = build_data(test_path)
y_true = []
for data in datas:
    for tag in data[1]:
        y_true.append(tag)

# x_test = modify_data(datas)
# y_pre = []
# filename = open(test_path.replace(".txt","")+"_Result.txt", 'w+',encoding='utf-8')  
# for text in x_test:
#     string = build_input(text)
#     raw =  model.predict(string)[0]
#     tags = [label_dict[i] for i in raw][:len(text)]
#     for i,tag in enumerate(tags):
#         y_pre.append(tag)
#         filename.write(text[i]+'\t'+str(tag)+'\n') 
# filename.close()

datas2 = build_data('test/output.txt')
y_pre = []
for data in datas2:
    for tag in data[1]:
        y_pre.append(tag)

y_pre = output(y_pre)
y_true = output(y_true)

filename = open(test_path.replace(".txt","")+"_P.txt", 'w+',encoding='utf-8')  
for value in y_pre:  
     filename.write(str(value)+'\n') 
filename.close() 
filename = open('test/output.txt'.replace(".txt","")+"_T.txt", 'w+',encoding='utf-8')  
for value in y_true:  
     filename.write(str(value)+'\n') 
filename.close()  
f = open("score.txt", 'w+', encoding='utf-8')

c = [x for x in y_pre if x not in y_true]
d = [y for y in y_true if y not in y_pre]
TP = len(y_pre)-len(c)
FP = len(c)
FN = len(d)
precision_score = TP/(TP+FP)
print("precision_score=TP/(TP+FP):"+str(precision_score))
recall_score = TP/(TP+FN)
print("recall_score=TP/(TP+FN):"+str(recall_score))
f1 = precision_score*recall_score * \
    2/(precision_score+recall_score)
print(
    "F1=precision_score*recall_score*2/(precision_score+recall_score):"+str(f1))
f.write(str(TP)+'\t'+str(FP)+'\t'+str(FN)+'\t'+str(precision_score) + '\t' +str(recall_score) + '\t' + str(f1) + '\n')
f.close()