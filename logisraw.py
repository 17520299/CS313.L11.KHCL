from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import precision_recall_fscore_support, classification_report

X_train,X_test,y_train,y_test=pickle.load(open('C:/Users/Admin/PycharmProjects/pythonProject/data_split.pkl','rb'))
logis = LogisticRegression().fit(X_train,y_train)
pickle.dump(logis,open('logisraw.pkl','wb'))

