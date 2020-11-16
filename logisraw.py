from sklearn.linear_model import LogisticRegression
import pickle

X_train,X_test,y_train,y_test=pickle.load(open('C:/Users/Admin/PycharmProjects/pythonProject/data_split.pkl','rb'))
logis = LogisticRegression().fit(X_train,y_train)
print(logis.score(X_test,y_test))
pickle.dump(logis,open('logisraw.pkl','wb'))

