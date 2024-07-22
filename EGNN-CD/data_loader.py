import pandas as pd
import torch.nn as nn
import torch

class Data_loader(nn.Module):
    def __init__(self,filename,know_num):
        self.know_num=know_num
        self.batch_size = 32
        self.ptr = 0

        super(Data_loader, self).__init__()
        filename = 'data/'+filename + '.csv'
        self.data = pd.read_csv(filename)

    def next_batch(self):
        if self.is_end():
            return None, None, None
        user_list, exer_list, label_list = [], [], []
        for count in range(self.batch_size):
            log = self.data.iloc[self.ptr + count]
            user_list.append(log['user_id'] - 1)
            exer_list.append(log['exer_id'] - 1)
            label_list.append(log['score'])

            # lst = [0.] * self.know_num
            # for k in eval(log['knowledge_code']):
            #     lst[k - 1] = 1.0
            # know_list.append(lst)
        self.ptr = self.ptr + self.batch_size
        return torch.LongTensor(user_list), torch.LongTensor(exer_list), torch.LongTensor(label_list)

    def is_end(self):
        if self.ptr + self.batch_size > len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0

