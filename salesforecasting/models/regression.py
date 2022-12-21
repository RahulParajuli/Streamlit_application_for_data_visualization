#%%
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import os
#%%
def saveModel(extractedModel, modelName):
    joblib.dump(extractedModel, modelName)

#%%
def preprocess(dataset, x_iloc_list, y_iloc, testSize):
    X = dataset.iloc[:, x_iloc_list].values
    y = dataset.iloc[:, y_iloc].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = 0)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = 0)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test
#%%
class regression:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def linear_regression(self):
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(self.X_train, self.y_train)
        y_pred = regressor.predict(self.X_test)
        saveModel(regressor, 'extractedModel/linear_regression.pkl')
        return y_pred
    
    def polynomial_regression(self):
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        poly_reg = PolynomialFeatures(degree = 4)
        X_poly = poly_reg.fit_transform(self.X_train)
        poly_reg.fit(X_poly, self.y_train)
        lin_reg_2 = LinearRegression()
        lin_reg_2.fit(X_poly, self.y_train)
        y_pred = lin_reg_2.predict(poly_reg.fit_transform(self.X_test))
        saveModel(lin_reg_2, 'extractedModel/polynomial_regression.pkl')
        return y_pred
    
    def support_vector_regression(self):
        from sklearn.svm import SVR
        regressor = SVR(kernel = 'rbf')
        regressor.fit(self.X_train, self.y_train)
        y_pred = regressor.predict(self.X_test)
        saveModel(regressor, 'extractedModel/support_vector_regression.pkl')
        return {"Predicted": y_pred}
    
    def grid_search(self, regressor, parameters):
        from sklearn.model_selection import GridSearchCV
        grid_search = GridSearchCV(estimator = regressor,
                                   param_grid = parameters,
                                   scoring = 'accuracy',
                                   cv = 10,
                                   n_jobs = -1)
        grid_search = grid_search.fit(self.X_train, self.y_train)
        best_accuracy = grid_search.best_score_
        best_parameters = grid_search.best_params_
        return {{"Best Accuracy":best_accuracy}, {"Best Parameter":best_parameters}}
    
    def k_fold_cross_validation(self, regressor):
        from sklearn.model_selection import cross_val_score
        accuracies = cross_val_score(estimator = regressor, X = self.X_train, y = self.y_train, cv = 10)
        return {{"Mean Accuracy":accuracies.mean()}, {"standard deviation accuracy":accuracies.std()}}
    
    def decision_tree(self):
        from sklearn.tree import DecisionTreeRegressor
        regressor = DecisionTreeRegressor(random_state = 0)
        regressor.fit(self.X_train, self.y_train)
        y_pred = regressor.predict(self.X_test)
        saveModel(regressor, 'extractedModel/decision_tree.pkl')
        return {"Predicted": y_pred}
    
    def random_forest(self):
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
        regressor.fit(self.X_train, self.y_train)
        y_pred = regressor.predict(self.X_test)
        saveModel(regressor, 'extractedModel/random_forest.pkl')
        return {"Predicted": y_pred}
    
    def xgboost(self):
        from xgboost import XGBRegressor
        regressor = XGBRegressor()
        regressor.fit(self.X_train, self.y_train)
        y_pred = regressor.predict(self.X_test)
        saveModel(regressor, 'extractedModel/xgboost.pkl')
        return {"Predicted": y_pred}
    
    def ANN(self):
        from keras.models import Sequential
        from keras.layers import Dense
        regressor = Sequential()
        regressor.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
        regressor.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
        regressor.add(Dense(units = 1, kernel_initializer = 'uniform'))
        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
        regressor.fit(self.X_train, self.y_train, batch_size = 10, epochs = 100)
        y_pred = regressor.predict(self.X_test)
        saveModel(regressor, 'extractedModel/ANN.pkl')
        return {"Predicted": y_pred}
    
    def ANN_tuning(self, regressor):
        from keras.wrappers.scikit_learn import KerasRegressor
        from keras.models import Sequential, Dense
        from sklearn.model_selection import GridSearchCV
        def build_regressor():
            regressor = Sequential()
            regressor.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
            regressor.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
            regressor.add(Dense(units = 1, kernel_initializer = 'uniform'))
            regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
            return regressor
        regressor = KerasRegressor(build_fn = build_regressor)
        parameters = {'batch_size': [25, 32],
                      'epochs': [100, 500]}
        grid_search = GridSearchCV(estimator = regressor,
                                   param_grid = parameters,
                                   scoring = 'neg_mean_squared_error',
                                   cv = 10)
        grid_search = grid_search.fit(self.X_train, self.y_train)
        best_parameters = grid_search.best_params_
        best_accuracy = grid_search.best_score_
        return {{"Best Accuracy":best_accuracy}, {"Best Parameter":best_parameters}}
    
    def accuracy(self, y_pred):
        from sklearn.metrics import r2_score
        return {"R2 Score":r2_score(self.y_test, y_pred)}
    
    def classification_report_plot(self, report):
        import seaborn as sns
        import matplotlib.pyplot as plt
        report_data = []
        lines = report.split('\n')
        for line in lines[2:-3]:
            row = {}
            row_data = line.split('      ')
            row['class'] = row_data[0]
            row['precision'] = float(row_data[1])
            row['recall'] = float(row_data[2])
            row['f1_score'] = float(row_data[3])
            row['support'] = float(row_data[4])
            report_data.append(row)
        df = pd.DataFrame.from_dict(report_data)
        df = df.set_index('class')
        df = df.drop('avg / total')
        df.plot(kind='bar', figsize=(15, 10))
        plt.show()
        return df
    
    def confusion_matrix_plot(self, cm):
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True)
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.show()
    
    def roc_curve_plot(self, y_pred):
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred)
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, color='orange', label='ROC')
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()
    
    def precision_recall_curve_plot(self, y_pred):
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(self.y_test, y_pred)
        plt.figure(figsize=(5, 5))
        plt.plot(recall, precision, color='orange', label='Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.show()
    
    def k_fold_cross_validation(self, regressor):
        from sklearn.model_selection import cross_val_score
        accuracies = cross_val_score(estimator = regressor, X = self.X_train, y = self.y_train, cv = 10)
        return {"Accuracy":accuracies.mean(), "Standard Deviation":accuracies.std()}
    
    def grid_search(self, regressor, parameters):
        from sklearn.model_selection import GridSearchCV
        grid_search = GridSearchCV(estimator = regressor,
                                   param_grid = parameters,
                                   scoring = 'neg_mean_squared_error',
                                   cv = 10)
        grid_search = grid_search.fit(self.X_train, self.y_train)
        best_parameters = grid_search.best_params_
        best_accuracy = grid_search.best_score_
        return {{"Best Accuracy":best_accuracy}, {"Best Parameter":best_parameters}}
    
    def random_search(self, regressor, parameters):
        from sklearn.model_selection import RandomizedSearchCV
        random_search = RandomizedSearchCV(estimator = regressor,
                                           param_distributions = parameters,
                                           n_iter = 10,
                                           scoring = 'neg_mean_squared_error',
                                           cv = 10)
        random_search = random_search.fit(self.X_train, self.y_train)
        best_parameters = random_search.best_params_
        best_accuracy = random_search.best_score_
        return {{"Best Accuracy":best_accuracy}, {"Best Parameter":best_parameters}}