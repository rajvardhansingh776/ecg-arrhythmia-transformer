from sklearn.metrics import accuracy_score,f1_score,confusion_matrix

def evaluate(preds,labels):

    acc=accuracy_score(labels,preds)

    f1=f1_score(labels,preds,average="macro")

    cm=confusion_matrix(labels,preds)

    return acc,f1,cm