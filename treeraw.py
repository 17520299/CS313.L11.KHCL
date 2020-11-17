from sklearn.tree import DecisionTreeClassifier
import pickle

X_train, X_test, y_train, y_test = pickle.load(open('C:\\Users\\Admin\\PycharmProjects\\CS313.L11.KHCL\\data_split.pkl','rb'))
tree = DecisionTreeClassifier(criterion='entropy',max_depth=12).fit(X_train,y_train)

pickle.dump(tree,open('treeraw.pkl','wb'))