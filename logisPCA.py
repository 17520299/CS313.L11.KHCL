from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import pickle

X_train,X_test,y_train,y_test=pickle.load(open('C:/Users/Admin/PycharmProjects/pythonProject/data_split.pkl','rb'))
pca = PCA(n_components=16)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.fit_transform(X_test)
logis = LogisticRegression().fit(X_train_pca,y_train)
pickle.dump(logis,open('logisPCA.pkl','wb'))

