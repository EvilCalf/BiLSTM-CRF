import os
import pandas as pd


class TransferData:
    def __init__(self):
        self.label_dict = {
            '疾病和诊断': 'DISEASE',
            '影像检查': 'TESTPROC',
            '实验室检验': 'TESTLAB',
            '手术': 'TREATMENT',
            '解剖部位': 'BODY',
            '药物': 'DRUGS',
        }

        self.cate_dict = {
            'O': 0,
            'B-DISEASE': 1,
            'I-DISEASE': 2,
            'B-TESTPROC': 3,
            'I-TESTPROC': 4,
            'B-TESTLAB': 5,
            'I-TESTLAB': 6,
            'B-BODY': 7,
            'I-BODY': 8,
            'B-DRUGS': 9,
            'I-DRUGS': 10,
            'B-TREATMENT': 11,
            'I-TREATMENT': 12,
        }
        self.entity_dirpath = "data/yidu/test"
        self.train_filepath = "train/yidu_test.txt"
        return

    def transfer(self):
        f = open(self.train_filepath, 'w+', encoding='utf-8')
        for root, dirs, files in os.walk(self.entity_dirpath):
            for file in files:
                json_path = root+"/"+file
                data = pd.read_json(json_path)
                if data.size == 0:
                    continue
                res_dict = {}
                content = data["originalText"][0]
                for i in enumerate(data["entities"]):
                    start = int(i[1]['start_pos'])
                    end = int(i[1]['end_pos'])
                    label = i[1]["label_type"]
                    label_id = self.label_dict.get(label)
                    for i in range(start, end):
                        if i == start:
                            label_cate = 'B-'+label_id
                        else:
                            label_cate = 'I-'+label_id
                        res_dict[i] = label_cate
                for indx, char in enumerate(content):
                    char_label = res_dict.get(indx, 'O')
                    word_list = ['。']
                    if char in word_list:
                        char_label = 'O'
                    if char != ' ':
                        f.write(char + '\t' + char_label + '\n')
                print("%s 完成！" % json_path)
        f.close()
        return


if __name__ == '__main__':
    handler = TransferData()
    train_datas = handler.transfer()
