from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from types import GeneratorType
import numpy as np
import pandas as pd
import numpy as np
import cv2
import os

PATH = './'

# Forma uma matriz de confusão respondendo a pergunta: é pulmão?
def confusion_matrix(ref, pred):

  matrix = []
  for i in range(7):
    matrix.append([])
    for j in range(7):
      matrix[i].append(0)

  for i in range(len(ref)):
      matrix[ref[i]][pred[i]] += 1

  tp = 0
  tn = 0
  fp = 0
  fn = 0
  for i in range(0,6):
    tp += matrix[i][i]

    for j in range(0,6):
        if i != j:
            fp += matrix[i][j]
            fn += matrix[j][i]
            for k in range(0,6): 
               if k != i:
                tn += matrix[j][k]

  print( tp, tn, fp, fn)
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

    # Retira a coluna label
    data_train = df_train.loc[:, df_train.columns != 'label']
    data_test = df_test.loc[:, df_test.columns != 'label']

    data_train.drop(columns='original', inplace=True)
    data_test.drop(columns='original', inplace=True)

    # Normaliza
    scaler = StandardScaler()
    data_train = scaler.fit_transform(data_train)
    data_test = scaler.transform(data_test)

    # Montém somente a coluna 'label'
    target_train = df_train['label']
    target_test = df_test ['label']

    return data_train, data_test, target_train, target_test


def main():

 est_list=[100,200,300,400]
 depth_list=[5,10,20,30,40,50]

 for est in est_list:
  for depth in depth_list:   
   ses = []
   esp = []
   f1 = []
   for i in range(1,6):

    # Carrega os dados
    data_train, data_test, target_train, target_test = process_data (i)

    # Carrega os classificadores
    RF = RandomForestClassifier(n_estimators=est, random_state=42, max_depth = depth)

    # Realiza o treinamento e as predições
    rf  = classifier(RF , data_train, data_test, target_train, target_test)

    ses.append(rf['sensibilidade'])
    esp.append(rf['especificidade'])
    f1.append(rf['f1'])

    # Salva os resultados
   with open('rf.csv', 'a+') as file:
    file.write(f"{est};{depth};{sum(ses)/len(ses)};{sum(esp)/len(esp)};{sum(f1)/len(f1)}\n")
        

main()
