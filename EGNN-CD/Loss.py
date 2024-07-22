import torch
import torch.nn as nn

Near_0=1e-8

class loss_function(nn.Module):
    def __init__(self):
        super(loss_function, self).__init__()
        self.loss_func=nn.NLLLoss()

    def forward(self,output_1,label,optimizer):
        loss=0
        temp=0
        output_0 = torch.ones(output_1.size()).to('cuda') - output_1
        output = torch.cat((output_0, output_1), 1)
        output=self.eliminate_0(output)
        loss=self.loss_func(torch.log(output),label)

        for i in optimizer.param_groups:
            for j in i['params']:
                # print(type(j.data), j.shape,j.data.dim())
                if j.data.dim() == 2:
                    temp += 0.002*torch.sqrt(j.data.pow(2).sum())                #torch.t()

        loss+=temp
        return loss

    def eliminate_0(self,inputs):
        for i in range(len(inputs)):
            for j in range(2):
                if inputs[i][j] < Near_0:
                    inputs[i][j] = Near_0
        return inputs