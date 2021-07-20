import os
import jsonlines


class TransferData:
    def __init__(self):
        self.cate_dict = {
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
        self.entity_dirpath = "data/sict"
        self.train_filepath = "train/sict_train.txt"
        return

    def transfer(self):
        f = open(self.train_filepath, 'w+', encoding='utf-8')
        for root, dirs, files in os.walk(self.entity_dirpath):
            for file in files:
                # with open(root+"/"+file, "r", encoding='utf-8') as f:
                #     reader = csv.reader(f)
                #     data = list(reader)
                # if len(data) == 0:
                #     continue
                data = []
                with open(root+"/"+file, "r+", encoding="utf8") as f1:
                    for item in jsonlines.Reader(f1):
                        data.append(item)
                for i, strs in enumerate(data):
                    res_dict = {}
                    for j, raw in enumerate(strs['label']):
                        start = int(raw[0])
                        end = int(raw[1])
                        label_id = raw[2]
                        for i in range(start, end):
                            if i == start:
                                label_cate = 'B-'+label_id
                            else:
                                label_cate = 'I-'+label_id
                            res_dict[i] = label_cate
                    for indx, char in enumerate(strs['data']):
                        char_label = res_dict.get(indx, 'O')
                        word_list = ['。']
                        if char in word_list:
                            char_label = 'O'
                        if char != ' ':
                            f.write(char + '\t' + char_label + '\n')
        f.close()
        return


if __name__ == '__main__':
    handler = TransferData()
    train_datas = handler.transfer()
