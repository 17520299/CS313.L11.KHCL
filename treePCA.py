from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
import pickle

pca = PCA(n_components=23)
X_train,X_test,y_train,y_test = pickle.load(open('C:/Users/Admin/PycharmProjects/pythonProject/data_split.pkl','rb'))

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

tree = DecisionTreeClassifier(criterion='entropy',max_depth=12).fit(X_train_pca,y_train)
pickle.dump(tree,open('treePCA.pkl','wb'))

