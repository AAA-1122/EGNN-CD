import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import sys
from sklearn.metrics import roc_auc_score, mean_absolute_error
from data_loader import Data_loader
from model import Cat_model
# from Loss import loss_function
# from predict import test
from Evaluate import eval_all


exer_num = 20
know_num = 8
user_num = 536


epochs=200
Near_0=1e-8
lr=0.008
exp_num=128
fold=1
#1e-45


def eliminate_0(inputs):
    for i in range(len(inputs)):
        for j in range(2):
            if inputs[i][j]<Near_0:
                inputs[i][j]=Near_0
    return inputs

def load_snapshot(model, filename):
    f = open(filename, 'rb')
    model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
    f.close()


def train(i,lr):
    # i=1
    data_loader = Data_loader('train_%s'%str(i+1), know_num)
    net = Cat_model(user_num=user_num, exer_num=exer_num, know_num=know_num, exp_num=exp_num).to('cuda')
    # load_snapshot(net, 'model/model_0.003epoch_' + str(146))

    optimizer = optim.Adam(net.parameters(), lr=lr)
    print('training model...')
    loss_func = nn.NLLLoss()

    for epoch in range(epochs):

        data_loader.reset()
        running_loss = 0.0
        batch_count = 0
        while not data_loader.is_end():
            batch_count = batch_count + 1
            input_stu_ids, input_exer_ids, labels = data_loader.next_batch()
            input_stu_ids, input_exer_ids, labels = input_stu_ids.to('cuda'), input_exer_ids.to('cuda'), labels.to(
                'cuda')
            optimizer.zero_grad()
            output_1 = net.forward(input_stu_ids, input_exer_ids)

            output_0 = torch.ones(output_1.size()).to('cuda') - output_1
            output = torch.cat((output_0, output_1), 1)
            output=eliminate_0(output)


            # grad_penalty = 0
            loss = loss_func(torch.log(output), labels,)  #
            loss.backward()
            optimizer.step()
            net.apply_clipper()

            running_loss = running_loss + loss.item()
            if batch_count % 200 == 199:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_count + 1, running_loss / 200))
                running_loss = 0.0

        # validate and save current model every epoch
        # validate(net, epoch,k=i)
        tt(net, epoch,k=i)
        save_snapshot(net, 'model/model_' +str(lr)+'epoch_'+ str(epoch + 1))



def validate(model, epoch,k):
    data_loader = Data_loader('valid_%s'%str(k+1),know_num)
    net = Cat_model(user_num=user_num,exer_num=exer_num,know_num=know_num,exp_num=exp_num)
    print('validating model...')
    data_loader.reset()
    # load model parameters
    net.load_state_dict(model.state_dict())
    net = net.to('cuda')
    net.eval()

    batch_count, batch_avg_loss = 0, 0.0
    pred_all, label_all = [], []
    while not data_loader.is_end():
        batch_count =batch_count+ 1
        input_stu_ids, input_exer_ids, labels = data_loader.next_batch()
        input_stu_ids, input_exer_ids, labels = input_stu_ids.to('cuda'), input_exer_ids.to('cuda'), labels.to('cuda')
        output = net.forward(input_stu_ids, input_exer_ids)
        output = output.view(-1)

        pred_all =pred_all+ output.to(torch.device('cpu')).tolist()
        label_all =label_all+ labels.to(torch.device('cpu')).tolist()


    Acc,AUPR,AUC,F1,Pre,Rec=eval_all(pred=pred_all,label=label_all)
    print('epoch= %d, accuracy= %f, aupr= %f, auc= %f, f1= %f, pre= %f, rec= %f' % (epoch+1, Acc,AUPR,AUC,F1,Pre,Rec))
    with open('result/model_val_%s.txt'%str(i+1), 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, aupr= %f, auc= %f, f1= %f, pre= %f, rec= %f\n' % (epoch+1, Acc,AUPR,AUC,F1,Pre,Rec))

    return Acc,AUPR,AUC,F1,Pre,Rec
def tt(model,epoch,k):
    data_loader = Data_loader('test_%s'%str(k+1),know_num)
    net = Cat_model(user_num=user_num,exer_num=exer_num,know_num=know_num,exp_num=exp_num)
    print('testing model...')
    data_loader.reset()
    # load model parameters
    net.load_state_dict(model.state_dict())
    # load_snapshot(net, 'model/model_0.003epoch_' + str(epoch))
    net = net.to('cuda')
    net.eval()

    correct_count, exer_count = 0, 0
    batch_count, batch_avg_loss = 0, 0.0
    pred_all, label_all = [], []

    while not data_loader.is_end():
        batch_count = batch_count + 1
        input_stu_ids, input_exer_ids, labels = data_loader.next_batch()
        input_stu_ids, input_exer_ids, labels = input_stu_ids.to('cuda'), input_exer_ids.to('cuda'), labels.to('cuda')
        output = net.forward(input_stu_ids, input_exer_ids)
        output = output.view(-1)

        # # compute accuracy
        # for i in range(len(labels)):
        #     if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
        #         correct_count =correct_count+ 1
        # exer_count =exer_count+ len(labels)
        pred_all =pred_all+ output.to(torch.device('cpu')).tolist()
        label_all =label_all+ labels.to(torch.device('cpu')).tolist()



    Acc, AUPR, AUC, F1, Pre, Rec = eval_all(pred=pred_all, label=label_all)
    print('epoch= %d, accuracy= %f, aupr= %f, auc= %f, f1= %f, pre= %f, rec= %f' % (
    epoch + 1, Acc, AUPR, AUC, F1, Pre, Rec))
    with open('result/model_test_%s.txt'%str(k+1), 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, aupr= %f, auc= %f, f1= %f, pre= %f, rec= %f\n' % (
        epoch + 1, Acc, AUPR, AUC, F1, Pre, Rec))

    return Acc, AUPR, AUC, F1, Pre, Rec

def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()


if __name__ == '__main__':
    # lr_list = []

    # device = torch.device(('cuda:0') if torch.cuda.is_available() else 'cpu')

    for i in range(fold):
        train(i=i,lr=lr)
