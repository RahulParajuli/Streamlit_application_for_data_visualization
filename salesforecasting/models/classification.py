#%%
from re import S
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import os

#%%
#function to save model
def saveModel(extractedModel, modelName):
    joblib.dump(extractedModel, modelName)

#%%
#for class based model
class classification:
    def __init__ (self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def accuracy(self, confusion_matrix):
        sum, total = 0,0
        for i in range(len(confusion_matrix)):
            for j in range(len(confusion_matrix[i])):
                if i == j:
                    sum += confusion_matrix[i][j]
                total += confusion_matrix[i][j]
        return sum/total
    
    def classification_report_plot(self, clf_report):
        clf_report = pd.DataFrame(clf_report).transpose()
        clf_report.plot(kind='bar',figsize=(10,10))
        plt.show()

    def LR(self):
        from sklearn.linear_model import LogisticRegression
        lr_classifier = LogisticRegression(random_state = 0)
        lr_classifier.fit(self.X_train, self.y_train)
        y_pred = lr_classifier.predict(self.X_test)
        saveModel(lr_classifier, 'extractedModel/lr_classifier.pkl')
        cm = confusion_matrix(self.y_test, y_pred)
        return {{"Accuracy: ": self.accuracy(cm)}, {"classification_report":classification_report(self.y_test, y_pred)},lr_classifier}
        
    def KNN(self):
        from sklearn.neighbors import KNeighborsClassifier
        knn_classifier = KNeighborsClassifier()
        knn_classifier.fit(self.X_train, self.y_train)
        y_pred = knn_classifier.predict(self.X_test)
        saveModel(knn_classifier, 'extractedModel/knn_classifier.pkl')
        cm = confusion_matrix(self.y_test, y_pred)
        return {{"Accuracy: ": self.accuracy(cm)}, {"classification_report":classification_report(self.y_test, y_pred)},knn_classifier}
    
    def SVM(self, kernel_type):
        from sklearn.svm import SVC
        svm_classifier = SVC(kernel = kernel_type)
        svm_classifier.fit(self.X_train, self.y_train)
        y_pred = svm_classifier.predict(self.X_test)
        saveModel(svm_classifier, 'extractedModel/svm_classifier.pkl')
        cm = confusion_matrix(self.y_test, y_pred)
        return {"Accuracy: ": self.accuracy(cm)}, {"classification_report":classification_report(self.y_test, y_pred)},svm_classifier

    def NB(self):
        from sklearn.naive_bayes import GaussianNB
        nb_classifier = GaussianNB()
        nb_classifier.fit(self.X_train, self.y_train)
        y_pred = nb_classifier.predict(self.X_test)
        saveModel(nb_classifier, 'extractedModel/nb_classifier.pkl')
        cm = confusion_matrix(self.y_test, y_pred)
        return {"Accuracy: ": self.accuracy(cm)}, {"classification_report":classification_report(self.y_test, y_pred)},nb_classifier
    
    
    def DT(self):
        from sklearn.tree import DecisionTreeClassifier
        tree_classifier = DecisionTreeClassifier()
        tree_classifier.fit(self.X_train, self.y_train)
        y_pred = tree_classifier.predict(self.X_test)
        SaveModel(tree_classifier, 'extractedModel/tree_classifier.pkl')
        cm = confusion_matrix(self.y_test, y_pred)
        return {"Accuracy: ": self.accuracy(cm)}, {"classification_report":classification_report(self.y_test, y_pred)},tree_classifier
        
    
    def RF(self):
        from sklearn.ensemble import RandomForestClassifier
        rf_classifier = RandomForestClassifier()
        rf_classifier.fit(self.X_train, self.y_train)
        y_pred = rf_classifier.predict(self.X_test)
        saveModel(rf_classifier, 'extractedModel/rf_classifier.pkl')
        cm = confusion_matrix(self.y_test, y_pred)
        return {"Accuracy: ": self.accuracy(cm)}, {"classification_report":classification_report(self.y_test, y_pred)},rf_classifier
        
    
    def XGB(self):
        from xgboost import XGBClassifier
        xgb_classifier = XGBClassifier()
        xgb_classifier.fit(self.X_train, self.y_train)
        y_pred = xgb_classifier.predict(self.X_test)
        saveModel(xgb_classifier, 'extractedModel/xgb_classifier.pkl')
        cm = confusion_matrix(self.y_test, y_pred)
        return {"Accuracy: ": self.accuracy(cm)}, {"classification_report":classification_report(self.y_test, y_pred)},xgb_classifier
        
    
    def ANN(self):
        from keras.models import Sequential
        from keras.layers import Dense
        classifier = Sequential()
        classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
        classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
        classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        classifier.fit(self.X_train, self.y_train, batch_size = 10, epochs = 100)
        y_pred = classifier.predict(self.X_test)
        saveModel(classifier, 'extractedModel/ann_classifier.pkl')
        y_pred = (y_pred > 0.5)
        cm = confusion_matrix(self.y_test, y_pred)
        return {"Accuracy: ": self.accuracy(cm)}, {"classification_report":classification_report(self.y_test, y_pred)},classifier
    
    def ANN_tuning(self, classifier):
        from keras.wrappers.scikit_learn import KerasClassifier
        from keras.models import Sequential, Dense
        from sklearn.model_selection import GridSearchCV
        def build_classifier(optimizer):
            classifier = Sequential()
            classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
            classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
            classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
            classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
            return classifier
        classifier = KerasClassifier(build_fn = build_classifier)
        parameters = {'batch_size': [25, 32],
                      'epochs': [100, 500],
                      'optimizer': ['adam', 'rmsprop']}
        grid_search = GridSearchCV(estimator = classifier,
                                   param_grid = parameters,
                                   scoring = 'accuracy',
                                   cv = 10)
        grid_search = grid_search.fit(self.X_train, self.y_train)
        best_parameters = grid_search.best_params_
        best_accuracy = grid_search.best_score_
        return best_parameters, best_accuracy