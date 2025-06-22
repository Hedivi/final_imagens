from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#from types import GeneratorType
import numpy as np
import pandas as pd
import numpy as np
import cv2
import os

PATH = './'

# Forma uma matriz de confusão e retorna os valores de verdadeiro positivo, ...
def confusion_matrix(ref, pred):

  # Forma uma matriz
  matrix = []
  for i in range(7):
    matrix.append([])
    for j in range(7):
      matrix[i].append(0)

  # Preenche a matriz com os valores reais x preditos
  for i in range(len(ref)):
      matrix[ref[i]][pred[i]] += 1

  # Calcula tp, fp, tn, fn
  tp = 0
  tn = 0
  fp = 0
  fn = 0
  for i in range(0,6):
    tp += matrix[i][i] # diagonal principal

    for j in range(0,6):
        if i != j:
            fp += matrix[i][j] # linha da classe
            fn += matrix[j][i] # coluna da classe
            for k in range(0,6): 
               if k != i:
                tn += matrix[j][k] # resto da matriz
 
  return tp, tn, fp, fn


# Realiza a classificação de acordo com o classificador passaod em FUNCTION
def classifier (function, data_train, data_test, target_train, target_test):

  # Treinamento do classificador
  function.fit(data_train, target_train)

  # Previsão do classificador
  target_pred = function.predict(data_test)

  # Calcula as métricas
  tp, tn, fp, fn = confusion_matrix(target_test, target_pred)

  sensibilidade = tp / (tp + fn)
  especificidade = tn / (tn + fp)
  f1 = 2 * (sensibilidade * especificidade) / (sensibilidade + especificidade)

  return {'sensibilidade': round(float(sensibilidade), 4), 'especificidade': round(float(especificidade), 4), 'f1': round(float(f1), 4)}


def process_data (count):

    # Carrega dados
    df_train = pd.read_csv(PATH + f'folds_stratified_groupk/characteristic_train_{str(count)}.csv')

    df_test = pd.read_csv(PATH + f'folds_stratified_groupk/characteristic_test_{str(count)}.csv')

    # Retira a coluna label e original
    data_train = df_train.loc[:, df_train.columns != 'label']
    data_test = df_test.loc[:, df_test.columns != 'label']

    data_train = data_train.drop(columns='original', inplace=False)
    data_test = data_test.drop(columns='original', inplace=False)

    # Normaliza
    scaler = StandardScaler()
    data_train = scaler.fit_transform(data_train)
    data_test = scaler.transform(data_test)

    # Montém somente a coluna 'label'
    target_train = df_train['label']
    target_test = df_test ['label']

    # Retorna os dados em dataframes pronto para treino e teste
    return data_train, data_test, target_train, target_test


def main():

  # Cross-validation com 5 folds
  for i in range(1,6):

    # Carrega os dados
    data_train, data_test, target_train, target_test = process_data (i)

    # Carrega os classificadores
    KNN = KNeighborsClassifier(n_neighbors=11)
    RF = RandomForestClassifier(n_estimators=300, random_state=42, max_depth = 30)
    SVM = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)

    # Realiza o treinamento e as predições
    knn = classifier(KNN, data_train, data_test, target_train, target_test)
    rf  = classifier(RF , data_train, data_test, target_train, target_test)
    svm = classifier(SVM, data_train, data_test, target_train, target_test)

    # Salva os resultados
    with open('results.csv', 'a+') as file:
        file.write(f"{str(i)};knn;{str(knn['sensibilidade'])};{str(knn['especificidade'])};{str(knn['f1'])}\n")
        file.write(f"{str(i)};rf ;{str(rf['sensibilidade'])};{str(rf['especificidade'])};{str(rf['f1'])}\n")
        file.write(f"{str(i)};svm;{str(svm['sensibilidade'])};{str(svm['especificidade'])};{str(svm['f1'])}\n\n")
        

main()
