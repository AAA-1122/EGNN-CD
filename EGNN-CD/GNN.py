import torch
import torch.nn as nn

# 4163x17746-------4163x123
class user_exer_gnn(nn.Module):
    def __init__(self,user_num,exer_num,know_num,exp_num):
        self.user_num=user_num
        self.exer_num=exer_num
        self.know_num=know_num
        self.exp_num=exp_num

        super(user_exer_gnn, self).__init__()

        # net
        # self.conv1=nn.Conv2d(in_channels=1,out_channels=1,kernel_size=5)
        self.linear1=nn.Linear(self.exer_num,2048)
        self.drop1=nn.Dropout(p=0.5)
        self.linear2=nn.Linear(2048,256)
        self.drop2=nn.Dropout(p=0.5)
        self.linear3=nn.Linear(256,self.exp_num)

    def forward(self,input):
        # x=self.conv1(inputs.unsqueeze(0).unsqueeze(0))
        x=self.drop1(torch.sigmoid(self.linear1(input)))
        x=self.drop2(torch.sigmoid(self.linear2(x)))
        output=torch.sigmoid(self.linear3(x))

        return output


# 4163x123-------4163x123
class user_know_gnn(nn.Module):
    def __init__(self, user_num,exer_num,know_num,exp_num):
        self.user_num = user_num
        self.know_num = know_num
        self.exer_num=exer_num
        self.exp_num=exp_num
        super(user_know_gnn, self).__init__()

        # net
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5)
        self.linear1 = nn.Linear(self.know_num, 512)
        self.drop1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, self.exp_num)


    def forward(self, input):
        # x = self.conv1(inputs.unsqueeze(0).unsqueeze(0))
        x = self.drop1(torch.sigmoid(self.linear1(input)))
        output = torch.sigmoid(self.linear2(x))

        return output

# 17746x4163-----17746x123
class exer_user_gnn(nn.Module):
    def __init__(self,user_num,exer_num,know_num,exp_num):
        self.user_num=user_num
        self.exer_num=exer_num
        self.know_num=know_num
        self.exp_num=exp_num
        super(exer_user_gnn, self).__init__()

        # net
        # self.conv1=nn.Conv2d(in_channels=1,out_channels=1,kernel_size=5)
        self.linear1=nn.Linear(self.user_num,1024)
        self.drop1=nn.Dropout(p=0.5)
        self.linear2=nn.Linear(1024,256)
        self.drop2=nn.Dropout(p=0.5)
        self.linear3=nn.Linear(256,self.exp_num)

    def forward(self,input):
        # x=self.conv1(inputs.unsqueeze(0).unsqueeze(0))
        x=self.drop1(torch.sigmoid(self.linear1(input)))
        x=self.drop2(torch.sigmoid(self.linear2(x)))
        output=torch.sigmoid(self.linear3(x))

        return output


# 17746x123----17746x123
class exer_know_gnn(nn.Module):
    def __init__(self,user_num,exer_num,know_num,exp_num):
        self.user_num=user_num
        self.exer_num=exer_num
        self.know_num=know_num
        self.exp_num=exp_num

        super(exer_know_gnn, self).__init__()

        # net
        # self.conv1=nn.Conv2d(in_channels=1,out_channels=1,kernel_size=5)
        self.linear1=nn.Linear(self.know_num,256)
        self.drop1=nn.Dropout(p=0.5)
        self.linear2=nn.Linear(256,self.exp_num)

    def forward(self,input):
        # x=self.conv1(inputs.unsqueeze(0).unsqueeze(0))
        x=self.drop1(torch.sigmoid(self.linear1(input)))
        output=torch.sigmoid(self.linear2(x))

        return output