from sklearn.ensemble import RandomForestClassifier
import pickle

X_train,X_test,y_train,y_test = pickle.load(open('C:/Users/Admin/PycharmProjects/pythonProject/data_split.pkl','rb'))

rdf = RandomForestClassifier(criterion='entropy',max_depth=12,n_estimators=10).fit(X_train,y_train)
#print(rdf.score(X_test,y_test))
pickle.dump(rdf,open('randomraw.pkl','wb'))