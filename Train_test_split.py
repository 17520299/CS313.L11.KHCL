import Preprocessing as pre
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
data = pd.read_csv('train.csv')
data = pre.ReadNDrop(data)
data,maxmA,maxmD = pre.OutlierRaw(data)
data = pre.Balance(data)
X ,Y =pre.Tranform(data)

X_train,X_test,y_train, y_test = train_test_split(X,Y,train_size=0.7, random_state=0)
print(y_train.value_counts())
print(y_test.value_counts())
pickle.dump([X_train,X_test,y_train, y_test],open('data_split.pkl','wb'))