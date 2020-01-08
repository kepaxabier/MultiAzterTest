import sys
import pandas as pd
import glob
import os
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
#####
import pickle

def main():
    menu()

def menu():
    create_directory('./dataset_aztertest_full')

    choice = input("""
                      1: Anadir label de clase a train y test dataset
                      2: Realizar 10-Fold CV con feature selection para obtener el mejor algoritmo y entrenarlo con train
                      3: Se carga el modelo y el selector características, se aplica el selector de atributos a los datos mediante el método transform() y se realiza la predicción utilizando el método predict()
                      4: Predict a file
                      5: Salir

                      Por favor, introduce la opción: """)

    if choice == "1":
        generar_train()
        generar_test()
    elif choice == "2":
        evaluate_and_obtain_best_algorithm()
    elif choice == "3":
        load_and_test()
    elif choice == "4":
        predictlabel()
    elif choice=="5":
        raise SystemExit(0)
    else:
        print("Debes seleccionar una de las cinco opciones (1, 2, 3 o 4)")
        print("Por favor, inténtalo de nuevo.")
    menu()

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def generar_train():
    final = None
    first = True
    levels = ['advanced', 'elementary', 'intermediate']
    for level in levels:
        level_subs = level[:3]
        if first:
            df = pd.read_csv("./Train/"+level_subs.capitalize()+"-Txt/results/full_results_aztertest.csv", delimiter=",")
            first = False
            final = df
            df['level'] = level
        else:
            df = pd.read_csv("./Train/"+level_subs.capitalize()+"-Txt/results/full_results_aztertest.csv", delimiter=",")
            df['level'] = level
            final = pd.concat([final, df], sort=False)
    final.to_csv("./dataset_aztertest_full/train_aztertest.csv", encoding='utf-8', index=False)

def generar_test():
    final = None
    first = True
    levels = ['advanced', 'elementary', 'intermediate']
    for level in levels:
        level_subs = level[:3]
        if first:
            df = pd.read_csv("./Test/"+level_subs.capitalize()+"-Txt/results/full_results_aztertest.csv", delimiter=",")
            first = False
            final = df
            df['level'] = level
        else:
            df = pd.read_csv("./Test/"+level_subs.capitalize()+"-Txt/results/full_results_aztertest.csv", delimiter=",")
            df['level'] = level
            final = pd.concat([final, df], sort=False)
    final.to_csv("./dataset_aztertest_full/test_aztertest.csv", encoding='utf-8', index=False)

def evaluate_and_obtain_best_algorithm():
    #Leer CSV
    dataset_train = pd.read_csv('./dataset_aztertest_full/train_aztertest.csv')
    dataset_test = pd.read_csv('./dataset_aztertest_full/test_aztertest.csv')

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
    #Seleccion de ATRIBUTOS
    #https://relopezbriega.github.io/blog/2016/04/15/ejemplo-de-machine-learning-con-python-seleccion-de-atributos/
    best_accuracy = 0.0
    # Utilizamos chi-cuadrado para seleccionar los K mejores atributos
    # En Python, para realizar la selección de atributos, se utiliza el método SelectKBest() de la librería SKLearn. En concreto, se va a utilizar la prueba de Chi cuadrado (χ2) para encontrar los mejores atributos y, en este caso, se establece que se desea conservar los 30 mejores (k=25). Una vez se hayan encontrado los 25 mejores atributos, se procede a transformar los conjuntos de entrenamiento y de pruebas.
    k_list = [25] #, 50, 75, 100]
    for k in k_list:
        #El algoritmo va a aplicar chi2 a todos los atributos y va a seleccionar los k que mejor resultado obtuvieron.
        selector=SelectKBest(score_func=chi2,k=k)
        # # Aplicamos el selector a los atributos (X) y obtenemos los K mejores (X_new)
        X_train_new = selector.fit_transform(X_train,y_train)
        X_test_new = selector.transform(X_test)
        # # Imprimimos los K mejores atributos junto con sus f-scores
        names = X_train.columns.values[selector.get_support()]
        #print(names)
        scores = selector.scores_[selector.get_support()]
        #print(scores)
        names_scores = list(zip(names, scores))
        #print(names_scores)
        #https://stackoverflow.com/questions/39839112/the-easiest-way-for-getting-feature-names-after-running-selectkbest-in-scikit-le
        ns_df = pd.DataFrame(data = names_scores, columns=['Feat_names', 'F_Scores'])
        # Sort the dataframe for better visualization
        ns_df_sorted = ns_df.sort_values(['F_Scores', 'Feat_names'], ascending = [False, True])
        print(ns_df_sorted)

        
        #seleccionar los algoritmos que mejor se adapten a las necesidades y, en segundo lugar, encontrar los parámetros más óptimos para dichos algoritmos de aprendizaje automático. Estos parámetros son llamados hiperparámetros, y sirven para controlar el proceso de aprendizaje de dichos algoritmos (Agrawal, 2019:https://towardsdatascience.com/hyperparameters-in-deep-learning-927f7b2084dd).Para seleccionar los algoritmos de aprendizaje, se hace uso del cheat-sheet que proporciona SKLearn(https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html) . El cheat-sheet de Scikit-Learn indica que los algoritmos que mejor se adaptan a los conjuntos de datos del proyecto. Para dichos algoritmos, se establecen los hiperparámetros que cada uno de los algoritmos procesará en el momento del entrenamiento y, el método GridSearchCV() de SKLearn será el encargado de encontrar los parámetros más óptimos que mejores resultados ofrezca. Concretamente, GridSearchCV() utiliza la técnica de K-Fold Cross Validation (con K=10).scikit-learn. (s.f.). sklearn.ensemble.AdaBoostClassifier. Obtenido de scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html ; scikit-learn. (s.f.). sklearn.ensemble.GradientBoostingClassifier. Obtenido de scikit-learn.com: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html ;scikit-learn. (s.f.). sklearn.ensemble.RandomForestClassifier. Obtenido de scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        # Initialize the pipeline with any estimator
        pipe = Pipeline(steps=[('estimator', SVC())])
        # Add a dict of estimator and estimator related parameters in this list
        params_grid = [#{'estimator':[KNeighborsClassifier()],
    			#'estimator__n_neighbors': [3,4,5,6,7,8,9,10,20],
    			#'estimator__weights': ['uniform', 'distance'],
    			#'estimator__algorithm': ['ball_tree', 'kd_tree', 'brute'],
    			#'estimator__metric': ['euclidean', 'manhattan', 'minkowski'],
    			#}
			#,
    			#{
    			# 'estimator': [LinearSVC()],
    			#'estimator__C': [0.025, 0.5, 1, 10, 100, 1000],
    			#'estimator__max_iter': [2000, 3000, 4000, 5000, 6000, 7000, 8000]
    			#},
    			#{
    			#'estimator': [RandomForestClassifier()],
    			#'estimator__criterion': ['gini', 'entropy'],
    			#'estimator__n_estimators': [10, 15, 20, 25, 30, 100, 500, 1000],
    			#'estimator__max_features': ['auto', 'sqrt', 'log2'],
    			#},
    			#{
    			#'estimator': [AdaBoostClassifier()],
    			#'estimator__n_estimators': [10, 15, 20, 25, 30, 100, 500, 1000],
    			#'estimator__learning_rate': [.001, 0.01, .1],
    			#},
    			#{
    			#'estimator': [GradientBoostingClassifier()],
    			#"estimator__learning_rate": [0.15, 0.1, 0.05, 0.01, 0.005, 0.001],
    			#"estimator__max_depth": [3, 5, 8],
    			#"estimator__max_features": ["log2", "sqrt"],
    			#"estimator__n_estimators": [100, 250, 500, 750, 1000, 1250, 1500, 1750],
    			#}
                        {
    			'estimator': [GradientBoostingClassifier()],
    			"estimator__learning_rate": [0.15],
    			"estimator__max_depth": [8],
    			"estimator__max_features": ["log2"],
    			"estimator__n_estimators": [500],
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
        #Una vez el método GridSearchCV() encuentra el algoritmo de aprendizaje que mejores resultados proporcione, se procede a entrenar el modelo clf.fit(X_train_new, y_train) para evaluarlo utilizando el conjunto de pruebas.
        clf.fit(X_train_new, y_train)
        #la predicción sobre el conjunto de pruebas mediante el método predict()
        results = clf.cv_results_
        if clf.best_score_ * 100 >= best_accuracy:
            final_clf = clf
            final_selector = selector
            best_accuracy = clf.best_score_ * 100

        # Best parameter set
        print('Best parameters found:\n', clf.best_params_)
        print("\nAccuracy: "+ str(clf.best_score_ * 100))
        print("\nAccuracy values: "+ str(results['mean_test_accuracy']))
        #print("\nPrecision: "+str(np.mean(results['mean_test_precision']) * 100))
        #print("\nRecall: "+str(np.mean(results['mean_test_recall']) * 100))
        #print("\nF-measure: "+str(np.mean(results['mean_test_f1_score']) * 100))

        text_file = open("./dataset_aztertest_full/ResultadosAztertestFull"+str(k)+".txt", "w")
        text_file.write("The %s best features are: \n" % str(k))
        text_file.write(ns_df_sorted.to_string())
        text_file.write("\nBest parameters found:: %s \n" % str(clf.best_params_))
        text_file.write("\nAccuracy: "+ str(clf.best_score_ * 100)+"\n")
        text_file.write("\nAccuracy values: "+ str(results['mean_test_accuracy'])+"\n")
        #text_file.write("\nPrecision: "+str(np.mean(results['mean_test_precision']) * 100)+"\n")
        #text_file.write("\nRecall: "+str(np.mean(results['mean_test_recall']) * 100)+"\n")
        #text_file.write("\nF-measure: "+str(np.mean(results['mean_test_f1_score']) * 100)+"\n")
        text_file.close()

    # Guardamos el modelo en un fichero para utilizarlo cuando queramos:Para guardar el modelo en un fichero, de manera que más adelante éste pueda ser cargado y utilizado para predecir la complejidad de un texto, se utiliza el método dump(). Además de guardar el modelo, también se guarda el selector de atributos para que se pueda realizar la misma selección de atributos sobre los datos a predecir
    joblib.dump(final_clf, './dataset_aztertest_full/classifier_aztertest_best.pkl')
    pickle.dump(final_selector, open("./dataset_aztertest_full/selectorAztertestFullBest.pickle", "wb"))

def load_and_test():
    #se carga el modelo y el selector mediante el método load(), se aplica el selector de atributos a los datos mediante el método transform() y se realiza la predicción utilizando el método predict()
    dataset_test = pd.read_csv('./dataset_aztertest_full/test_aztertest.csv')
    # Elegimos todas las columnas menos la clase ("level")
    feature_names = dataset_test.columns.tolist()
    feature_names.remove("level")
    # Cogemos los atributos
    X_test = dataset_test[feature_names]
    # Cogemos las clases
    y_test = dataset_test['level']

    # Para cargarlo, simplemente hacer lo siguiente:
    clf = joblib.load('./dataset_aztertest_full/classifier_aztertest_best.pkl')

    with open("./dataset_aztertest_full/selectorAztertestFullBest.pickle", "rb") as f:
        selector = pickle.load(f)

    X_test_new = selector.transform(X_test)

    # Obtenemos las predicciones utilizando el set de TEST previamente creado
    y_pred = clf.predict(X_test_new)

    # Imprimimos la matriz de confusion y la precision (accuracy)
    unique_label = np.unique(y_test)
    print("\nConfusion matrix for Test Set:")
    print(pd.DataFrame(confusion_matrix(y_test, y_pred, labels=unique_label),
                       index=['Real:{:}'.format(x) for x in unique_label],
                       columns=['Predicted:{:}'.format(x) for x in unique_label]))
    accuracy = str(accuracy_score(y_test, y_pred) * 100)
    print("\nAccuracy: "+ accuracy)
    #print("\nPrecision: "+str(precision_score(y_test, y_pred, average="macro") * 100))
    #print("\nRecall: "+str(recall_score(y_test, y_pred, average="macro") * 100))
    #print("\nF-measure: "+str(f1_score(y_test, y_pred, average="macro") * 100))

    text_file = open("./dataset_aztertest_full/ResultadosAztertestFull_TEST.txt", "w")
    text_file.write("\nAccuracy achieved with the TEST SET: %s \n" % accuracy+"\n")
    text_file.write("\nConfusion matrix for Test Set:")
    text_file.write(str(pd.DataFrame(confusion_matrix(y_test, y_pred, labels=unique_label),
                       index=['Real:{:}'.format(x) for x in unique_label],
                       columns=['Predicted:{:}'.format(x) for x in unique_label]))+"\n")
    #text_file.write("\nPrecision: "+str(precision_score(y_test, y_pred, average="macro") * 100)+"\n")
    #text_file.write("\nRecall: "+str(recall_score(y_test, y_pred, average="macro") * 100)+"\n")
    #text_file.write("\nF-measure: "+str(f1_score(y_test, y_pred, average="macro") * 100)+"\n")
    text_file.close()

def predict_dificulty(data):
    feature_names = data.columns.tolist()
    X_test = data[feature_names]
    # Para cargarlo, simplemente hacer lo siguiente:
    clf = joblib.load('./dataset_aztertest_full/classifier_aztertest_best.pkl')
    with open("./dataset_aztertest_full/selectorAztertestFullBest.pickle", "rb") as f:
         selector = pickle.load(f)
    #se aplica el selector de atributos a los datos mediante el método transform() 
    X_test_new = selector.transform(X_test)
    #y se realiza la predicción utilizando el método predict()
    return clf.predict(X_test_new)

#def filetodf():
    #df = pd.read_csv("./Train/"+level_subs.capitalize()+"-Txt/results/full_results_aztertest.csv", delimiter=",")
#   df = pd.read_csv('./dataset_aztertest_full/test_aztertest.csv')
#   return df

#def predictlabel():
#   df = filetodf()
#   prediction = predict_dificulty(df)
#   print("prediction:",prediction)

main()
