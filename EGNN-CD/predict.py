import torch
from Evaluate import eval_all
from data_loader import Data_loader
from model import Cat_model



exer_num = 20
know_num = 8
user_num = 536

exp_num=16


def test_self(epoch):
    data_loader = Data_loader('test',know_num)
    net = Cat_model(user_num=user_num,exer_num=exer_num,know_num=know_num,exp_num=exp_num)
    print('testing model...')
    data_loader.reset()
    # load model parameters
    load_snapshot(net, 'model/model_0.005epoch_' + str(epoch))
    # net.load_state_dict('model/model_epoch'+str(epoch))
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
        #         correct_count = correct_count + 1
        # exer_count = exer_count + len(labels)
        pred_all = pred_all + output.to(torch.device('cpu')).tolist()
        label_all = label_all + labels.to(torch.device('cpu')).tolist()

    Acc, AUPR, AUC, F1, Pre, Rec = eval_all_hunxiao(pred=pred_all, label=label_all)
    print('epoch= %d, accuracy= %f, aupr= %f, auc= %f, f1= %f, pre= %f, rec= %f' % (
        epoch + 1, Acc, AUPR, AUC, F1, Pre, Rec))
    with open('result/math1_hunxiao', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, aupr= %f, auc= %f, f1= %f, pre= %f, rec= %f\n' % (
            epoch + 1, Acc, AUPR, AUC, F1, Pre, Rec))

    return Acc, AUPR, AUC, F1, Pre, Rec






if __name__ == '__main__':
    for i in range(0,epochs):
        test_self(i+1)
