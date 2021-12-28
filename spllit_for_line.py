import os
num = 1
with open('data/yidu/subtask1_test_set_with_answer.jsonl', encoding='utf-8') as reader:
    for index, line in enumerate(reader):
        f = open("data/yidu/test/"+str(num)+".json", 'w+', encoding='utf-8')
        f.write(line)
        num = num+1
