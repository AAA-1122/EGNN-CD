from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error, \
    roc_auc_score,precision_recall_curve,auc,accuracy_score
import numpy as np

# label = [0, 1, 0]
# pred = [0.2, 0.8, 0.8]

def eval_all(pred,label):
    pred_all = np.array(pred)
    label_all = np.array(label)

    precision, recall, _ = precision_recall_curve(label, pred)
    AUPR=auc(recall, precision)
    AUC = roc_auc_score(label_all, pred_all)

    for index,i in enumerate(pred):
        if i>0.5:
            pred[index]=1
        else:
            pred[index]=0

    Acc=accuracy_score(pred,label)
    Rec = recall_score(label, pred, average='binary')
    Pre = precision_score(label, pred, average='binary')
    F1 = f1_score(label, pred)

    return Acc,AUPR,AUC,F1,Pre,Rec
