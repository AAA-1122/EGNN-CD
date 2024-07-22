import pandas as pd
import torch
import torch.nn as nn
from GNN import user_exer_gnn,user_know_gnn,exer_user_gnn,exer_know_gnn

class Cat_model(nn.Module):
    def __init__(self,user_num,exer_num,know_num,exp_num):
        self.user_num=user_num
        self.exer_num=exer_num
        self.know_num=know_num
        self.exp_num=exp_num
        self.prednet_input_len = self.exp_num*4
        super(Cat_model, self).__init__()

        self.emb=nn.Embedding(2,1,max_norm=1)

        self.user_exer_gnn = user_exer_gnn(self.user_num, self.exer_num, self.know_num,self.exp_num)
        self.user_know_gnn = user_know_gnn(self.user_num, self.exer_num, self.know_num,self.exp_num)
        self.know_user_gnn = exer_user_gnn(self.user_num, self.exer_num, self.know_num,self.exp_num)
        self.know_exer_gnn = exer_know_gnn(self.user_num, self.exer_num, self.know_num,self.exp_num)


        self.user_exer, self.user_know, self.exer_know = self.data_preprocess()
        self.user_exer, self.user_know, self.exer_know = self.user_exer.to('cuda:0'), self.user_know.to('cuda:0') ,self.exer_know.to('cuda:0')

        # self.user_exer, self.user_know, self.know_user, self.know_exer = self.user_exer.to('cuda:0'), self.user_know.to('cuda:0'), self.know_user.to('cuda:0'), self.know_exer.to('cuda:0')
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable
        # network structure
        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, user_list, exer_list):
        self.user_exer_mat = self.user_exer_gnn(self.user_exer)
        self.user_know_mat = self.user_know_gnn(self.user_know)
        self.exer_user_mat = self.know_user_gnn(self.user_exer.transpose(0, 1))
        self.exer_know_mat = self.know_exer_gnn(self.exer_know)


        user_express = torch.cat([self.user_exer_mat[user_list],self.user_know_mat[user_list]],dim=1)

        exer_express = torch.cat([self.exer_user_mat[exer_list],self.exer_know_mat[exer_list]],dim=1)

        input_x = torch.cat([user_express,exer_express],dim=1)

        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output = torch.sigmoid(self.prednet_full3(input_x))     #+ e_guess

        return output

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def data_preprocess(self):
        user_exer, user_know, exer_know=torch.zeros(self.user_num,self.exer_num),torch.zeros(self.user_num,self.know_num), \
                                        torch.zeros(self.exer_num,self.know_num),
        data=pd.read_csv('data/total.csv')
        for _,row in data.iterrows():
            user=row['user_id']-1
            exer=row['exer_id']-1
            score=row['score']
            log=row['knowledge_code']

            user_exer[user][exer]=1.0
            if score:
                for k in eval(log):
                    user_know[user][k-1]=1.0
            for k in eval(log):
                exer_know[exer][k-1]=1.0

        return user_exer, user_know, exer_know


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)
