#Importing Libraries
import pandas as pd
import numpy as np
# Import the necessary libraries first
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
#####
import pickle

#Leer CSV
dataset_train = pd.read_csv('/home/julen/Dropbox/TFG/corpus/en/3levels/AztertestTrainTest/Train/train_aztertest_with_different_cohmetrix.csv')
dataset_test = pd.read_csv('/home/julen/Dropbox/TFG/corpus/en/3levels/AztertestTrainTest/Test/test_aztertest_with_different_cohmetrix.csv')

#Imprimir numero de filas y columnas
print('Shape of the train dataset: ' + str(dataset_train.shape))
#Mostrar 5 primeros elementos
print(dataset_train.head())
print('\n')
#Mostrar las 3 clases
print('Labels: ' + str(dataset_train['level'].unique()))
print('\n')
print(dataset_train.groupby('level').size())

# Elegimos todas las columnas menos la clase ("level")
feature_names = dataset_train.columns.tolist()
feature_names.remove("level")
# Cogemos los atributos
X_train = dataset_train[feature_names]
# Cogemos las clases
y_train = dataset_train['level']

#Imprimir numero de filas y columnas
print('Shape of the test dataset: ' + str(dataset_test.shape))
#Mostrar 5 primeros elementos
print(dataset_test.head())
print('\n')
#Mostrar las 3 clases
print('Labels: ' + str(dataset_test['level'].unique()))
print('\n')
print(dataset_test.groupby('level').size())

# Elegimos todas las columnas menos la clase ("level")
feature_names = dataset_test.columns.tolist()
feature_names.remove("level")
# Cogemos los atributos
X_test = dataset_test[feature_names]
# Cogemos las clases
y_test = dataset_test['level']


# Utilizamos chi-cuadrado para seleccionar los 30 mejores atributos
k=30
#selector=SelectKBest(score_func=chi2,k=k)
selector=SelectKBest(score_func=f_classif,k=k)
# # Aplicamos el selector a los atributos (X) y obtenemos los 30 mejores (X_new)
X_train_new = selector.fit_transform(X_train,y_train)
X_test_new = selector.transform(X_test)
#
# # Imprimimos los 30 mejores atributos junto con sus f-scores
names = X_train.columns.values[selector.get_support()]
scores = selector.scores_[selector.get_support()]
names_scores = list(zip(names, scores))
ns_df = pd.DataFrame(data = names_scores, columns=['Feat_names', 'F_Scores'])
# Sort the dataframe for better visualization
ns_df_sorted = ns_df.sort_values(['F_Scores', 'Feat_names'], ascending = [False, True])
print(ns_df_sorted)

# Just initialize the pipeline with any estimator you like
pipe = Pipeline(steps=[('estimator', SVC())])

# Add a dict of estimator and estimator related parameters in this list

params_grid = [{
                'estimator':[KNeighborsClassifier()],
                'estimator__n_neighbors': [3,4,5,6,7,8,9,10,20],
                'estimator__weights': ['uniform', 'distance'],
                'estimator__algorithm': ['ball_tree', 'kd_tree', 'brute'],
                'estimator__metric': ['euclidean', 'manhattan', 'minkowski'],
                },
                {
                 'estimator': [LinearSVC()],
                'estimator__C': [0.025, 0.5, 1, 10, 100, 1000],
                'estimator__max_iter': [2000, 3000, 4000, 5000, 6000, 7000, 8000]
                },
                {
                'estimator': [RandomForestClassifier()],
                'estimator__criterion': ['gini', 'entropy'],
                'estimator__n_estimators': [10, 15, 20, 25, 30, 100, 500, 1000],
                'estimator__max_features': ['auto', 'sqrt', 'log2'],
                },
                {
                'estimator': [AdaBoostClassifier()],
                'estimator__n_estimators': [10, 15, 20, 25, 30, 100, 500, 1000],
                'estimator__learning_rate': [.001, 0.01, .1],
                },
                {
                'estimator': [GradientBoostingClassifier()],
                "estimator__learning_rate": [0.15, 0.1, 0.05, 0.01, 0.005, 0.001],
                "estimator__max_depth": [3, 5, 8],
                "estimator__max_features": ["log2", "sqrt"],
                "estimator__n_estimators": [100, 250, 500, 750, 1000, 1250, 1500, 1750],
                }
              ]


# Utilizamos stratified k-fold. Con este metodo partimos el TRAINING SET en 10 trozos iguales (9 trozos correspondientes a
# TRAIN y uno a DEV) y, por cada trozo, se mantiene la proporcion de los valores de cada clase (Stratified).
# https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6
skfold = StratifiedKFold(n_splits = 10)

scoring = {'accuracy': 'accuracy',
           'precision': 'precision_macro',
           'recall': 'recall_macro',
           'f1_score': 'f1_macro'}

clf = GridSearchCV(pipe, params_grid, n_jobs=-2, cv=skfold,  verbose=5, scoring=scoring, refit='accuracy')
clf.fit(X_train, y_train)

results = clf.cv_results_
# Best parameter set
print('Best parameters found:\n', clf.best_params_)
print("\nAccuracy: "+ str(np.mean(results['mean_test_accuracy']) * 100))
print("\nAccuracy values: "+ str(results['mean_test_accuracy']))
print("\nPrecision: "+str(np.mean(results['mean_test_precision']) * 100))
print("\nRecall: "+str(np.mean(results['mean_test_recall']) * 100))
print("\nF-measure: "+str(np.mean(results['mean_test_f1_score']) * 100))

# Obtenemos las predicciones utilizando el set de TEST previamente creado
y_pred = clf.predict(X_test)

# Imprimimos la matriz de confusion y la precision (accuracy)
unique_label = np.unique(y_test)
print("\nConfusion matrix for Test Set:")
print(pd.DataFrame(confusion_matrix(y_test, y_pred, labels=unique_label),
                   index=['Real:{:}'.format(x) for x in unique_label],
                   columns=['Predicted:{:}'.format(x) for x in unique_label]))
accuracy = str(accuracy_score(y_test, y_pred) * 100)
print("\nAccuracy: "+ accuracy)

text_file = open("ResultadosAztertestWithDifferentCohMetrixK"+str(k)+".txt", "w")
text_file.write("The %s best features are: \n" % str(k))
text_file.write(ns_df_sorted.to_string())
text_file.write("\nBest parameters found:: %s \n" % str(clf.best_params_))
text_file.write("\nAccuracy: "+ str(np.mean(results['mean_test_accuracy']) * 100)+"\n")
text_file.write("\nAccuracy values: "+ str(results['mean_test_accuracy'])+"\n")
text_file.write("\nPrecision: "+str(np.mean(results['mean_test_precision']) * 100)+"\n")
text_file.write("\nRecall: "+str(np.mean(results['mean_test_recall']) * 100)+"\n")
text_file.write("\nF-measure: "+str(np.mean(results['mean_test_f1_score']) * 100)+"\n")
text_file.write("\nAccuracy achieve with the TEST SET: %s \n" % accuracy+"\n")
text_file.write("\nConfusion matrix for Test Set:")
text_file.write(str(pd.DataFrame(confusion_matrix(y_test, y_pred, labels=unique_label),
                   index=['Real:{:}'.format(x) for x in unique_label],
                   columns=['Predicted:{:}'.format(x) for x in unique_label])))
text_file.close()

# Guardamos el modelo en un fichero para utilizarlo cuando queramos
joblib.dump(clf, 'classifier_aztertest_adapted_different_cohmetrix_k'+str(k)+'.pkl')
#pickle.dump(selector, open("selectorCohMetrixFullK"+str(k)+".pickle", "wb"))