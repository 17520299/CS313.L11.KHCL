import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns

pca = PCA(n_components=17)

X_train,X_test,y_train,y_test = pickle.load(open('C:/Users/Admin/PycharmProjects/CS313.L11.KHCL/data_split.pkl','rb'))
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.fit_transform(X_test)

logis = pickle.load(open('C:/Users/Admin/PycharmProjects/CS313.L11.KHCL/logisraw.pkl','rb'))
logisPCA = pickle.load(open('C:/Users/Admin/PycharmProjects/CS313.L11.KHCL/logisPCA.pkl','rb'))
tree = pickle.load(open('C:/Users/Admin/PycharmProjects/CS313.L11.KHCL/treeraw.pkl','rb'))
treePCA = pickle.load(open('C:/Users/Admin/PycharmProjects/CS313.L11.KHCL/treePCA.pkl','rb'))
random = pickle.load(open('C:/Users/Admin/PycharmProjects/CS313.L11.KHCL/randomraw.pkl','rb'))
randomPCA = pickle.load(open('C:/Users/Admin/PycharmProjects/CS313.L11.KHCL/randomPCA.pkl','rb'))

list = [logis,tree,random]
list_pca = [logisPCA,treePCA,randomPCA]
list_kq_pca =[]
list_kq_raw = []
for i,j in zip(list,list_pca):
    list_kq_raw.append(i.score(X_test,y_test))
    list_kq_pca.append(j.score(X_test_pca,y_test))
algorithm = ['LR','DT','RF']

def Chart(model,model1,model2,X_test,y_test,algorithm):
    precision_logis,recall_logis,f1_logis,support_logis = precision_recall_fscore_support(y_test,model.predict(X_test))
    predict_tree,recall_tree,f1_tree,support_tree = precision_recall_fscore_support(y_test,model1.predict(X_test))
    predict_random, recall_random, f1_random, support_random = precision_recall_fscore_support(y_test,model2.predict(X_test))
    def Dodo(model1,model2,model3,algorithm):
        list0 = [model1[0],model2[0],model3[0]]
        list1 = [model1[1],model2[1],model3[1]]
        index = np.arange(3)
        width = 0.2
        plt.bar(index,list0,width=width,color='blue',label='Neutral')
        plt.bar(index+width, list1, width=width, color='green', label='Satisfied')
        plt.ylabel("Metrics")
        plt.ylim(0,1.1)
        plt.xticks(index+width/2,algorithm)
        plt.legend(loc=2)
    plt.figure(figsize=(12,6))
    plt.subplot(1,3,1)
    Dodo(precision_logis,predict_tree,predict_random,algorithm)
    plt.subplot(1,3,2)
    Dodo(recall_logis,recall_tree,recall_random,algorithm)
    plt.title("Metrics Chart Comparation(Precision - Recall - F1-Score)")
    plt.subplot(1,3,3)
    Dodo(f1_logis,f1_tree,f1_random,algorithm)
    plt.show()
Chart(logis,tree,random,X_test,y_test,algorithm)



