from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
import numpy as np
from sklearn.metrics import accuracy_score
import pickle

def classify(sp, sw, pl, pw, classfr):

    features_user = np.zeros((1, 4))

    features_user[0][0] = float(sp)
    features_user[0][1] = float(sw)
    features_user[0][2] = float(pl)
    features_user[0][3] = float(pw)

    if (classfr == "dtc"):
        classifier = pickle.load(open('./Models/DecisionTreeClassifier.sav', 'rb'))
    elif (classfr == "knc"):
        classifier = pickle.load(open('./Models/KNeighborsClassifier.sav', 'rb'))
    elif (classfr == "gnb"):
        classifier = pickle.load(open('./Models/GaussianNB.sav', 'rb'))
    else:
        return "Invalid classifier!"
    
    predictions=classifier.predict(features_user)

    # 0 Iris Setosa
    # 1 Iris Versicolour
    # 2 Iris Virginica

    if (predictions == 0):
        return "Iris Setosa"
    elif(predictions == 1):
        return "Iris Versicolour"
    elif(predictions == 2):
        return "Iris Virginica"

def train(mthd):
    iris = load_iris()

    features = iris.data
    labels = iris.target

    features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size=.33)
    classifier = mthd
    classifier.fit(features_train,labels_train)

    predictions_test = classifier.predict(features_test)
    acc_score = accuracy_score(labels_test,predictions_test)

    mthdstr = str(mthd)
    name = mthdstr.replace("(", "")
    name = name.replace(")", "")

    pickle.dump(classifier, open('./Models/'+name+'.sav', 'wb'))
    
    file = open("./ModelsAccScore/"+name+".txt","w")
    file.write(str(acc_score))
    file.close()

if __name__ == "__main__":
    train(tree.DecisionTreeClassifier())
    train(neighbors.KNeighborsClassifier())
    train(GaussianNB())

    #print(classify(6.3,2.9,5.6,1.8, 'GaussianNB'))