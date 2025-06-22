import numpy as np
from pydicom import dcmread
from radiomics import featureextractor
import pandas as pd
import numpy as np
import SimpleITK as sitk
import os

PATH = './' 

# Gera um arquivo com a máscara de Otsu
def generate_mask(path):

  # Define o nome do arquivo
  name = path.replace('.dcm', '_mask.nii')

  # Verifica se o arquivo já existe
  if os.path.exists(name):
    return name

  # Carregar a imagem 3D
  imagem_3d = sitk.ReadImage(path, sitk.sitkFloat32)

  # Aplicar threshold de Otsu (gera uma imagem binária 3D)
  otsu = sitk.OtsuThreshold(imagem_3d, 0, 1)

  # Salvar a imagem binária resultante
  sitk.WriteImage(otsu, name)

  return name


# Define o extrator
def extractor():

    extract = featureextractor.RadiomicsFeatureExtractor(PATH + 'params.yaml')

    return extract

# Extrai as características das imagens
def characteristics(df, extract):

    characteristics = []

    for i in range(len(df['original'])):
        img_path = df['original'][i]
        mask_path = df['mask'][i]
        label = df['label'][i]

        # Carrega imagem e máscara com SimpleITK
        img = sitk.ReadImage(img_path)
        mask = sitk.ReadImage(mask_path)

        # Converte para arrays e imprime as dimensões
        img_np = sitk.GetArrayFromImage(img)
        mask_np = sitk.GetArrayFromImage(mask)

        # Executa extração de características
        try:
            features = extract.execute(img_path, mask_path)
            features['label'] = label
            features['original'] = img_path
            characteristics.append(features)
        # Se não conseguir, armazena as características no arquivo
        except Exception as e:
            with open ('erros.txt', 'a') as file:
                file.write(f"{i};{df['original'][i]};{df['mask'][i]}\n")

    # Retorna dataframe com as características extraídas
    return pd.DataFrame(characteristics)


# Carrega os dados
def load_data(doc_path):

    first = True
    data = []
    with open(doc_path, "r") as doc:
        for line in doc:
            # Elimina o cabeçalho da consulta
            if first:
                first = False
                continue

            # Define as informações a partir do arquivo
            path = PATH + line.split(',')[1]
            mask = generate_mask(path)
            label = int(line.split(',')[2])

            data.append(dict(original=path, mask=mask, label=label))

    return pd.DataFrame(data)


# Realiza o processamento dos dados
def process_data(count):

    # Define o caminho
    train_path = PATH + f'folds_stratified_groupk/fold{str(count)}_train.csv'
    test_path = PATH + f'folds_stratified_groupk/fold{str(count)}_val.csv'

    # Carrega os dados
    df_general_train = load_data(train_path)
    df_general_test = load_data(test_path)

    # Extrai as características
    extract = extractor()
    df_train = characteristics(df_general_train, extract)
    df_test = characteristics(df_general_test, extract)

    # Retira informação extra
    col_rm = [ col for col in df_train.columns if 'diagnostics' in col ]
    df_train = df_train.drop(col_rm, axis=1)
    df_test = df_test.drop(col_rm, axis=1)

    # Retorna dataframes 
    return df_train, df_test


def main():

  for i in range(1,6):

    # Carrega e processa os dados
    df_train, df_test = process_data(i)

    # Cria csv's com as características extraídas
    with open (PATH + f"folds_stratified_groupk/characteristic_train_{str(i)}.csv", 'w+') as file:
      df_train.to_csv(file, index=False)

    with open (PATH + f"folds_stratified_groupk/characteristic_test_{str(i)}.csv", 'w+') as file:
      df_test.to_csv(file, index=False)



main()
