from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score, accuracy_score

def evaluate(y_true, y_pred):
    print('macro precision: ' + str(precision_score(y_true, y_pred, average='macro')))
    print('macro recall: ' + str(recall_score(y_true, y_pred, average='macro')))
    print('macro f1: ' + str(f1_score(y_true, y_pred, average='macro')))
    print('micro precision: ' + str(precision_score(y_true, y_pred, average='micro')))
    print('micro recall: ' + str(recall_score(y_true, y_pred, average='micro')))
    print('micro f1: ' + str(f1_score(y_true, y_pred, average='micro')))
    print('weighted precision: ' + str(precision_score(y_true, y_pred, average='weighted')))
    print('weighted recall: ' + str(recall_score(y_true, y_pred, average='weighted')))
    print('weighted f1: ' + str(f1_score(y_true, y_pred, average='weighted')))
    print('precision: ' + str(precision_score(y_true, y_pred)))
    print('recall: ' + str(recall_score(y_true, y_pred)))
    print('f1: ' + str(f1_score(y_true, y_pred)))
    print('balanced accuracy: ' + str(balanced_accuracy_score(y_true, y_pred)))
    print('accuracy: ' + str(accuracy_score(y_true, y_pred)))