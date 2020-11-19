from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.decomposition import PCA
pca = PCA(n_components=17)
X_train,X_test,y_train,y_test = pickle.load(open('C:/Users/Admin/PycharmProjects/pythonProject/data_split.pkl','rb'))
X_train_pca = pca.fit_transform(X_train)
rdf = RandomForestClassifier(criterion='entropy',max_depth=12,n_estimators=10).fit(X_train_pca,y_train)
pickle.dump(rdf,open('randomPCA.pkl','wb'))