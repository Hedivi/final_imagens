import pandas as pd
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

  # Retorna noma da imagem
  return name


# Carrega os dados
def load_data(doc_path):

    first = True
    data = []
    i = 0
    with open(doc_path, "r") as doc:
        for line in doc:
            # Pula a linha de cabeçalho
            if first:
                first = False
                continue

            # Armazena as informações necessárias para identificaçãoe extração de características das imagens
            path = PATH + line.split(',')[1]
            mask = generate_mask(path)
            label = int(line.split(',')[2])

            # Cria uma lista de dicionários com as informações
            data.append(dict(original=path, mask=mask, label=label))

    # Retorna um Dataframe com os dados
    return pd.DataFrame(data)


# Realiza o processamento dos dados
def process_data():

    # Define os arquivos csv das imagens
    train_path = PATH + f'folds_stratified_groupk/fold{str(count)}_train.csv'
    test_path = PATH + f'folds_stratified_groupk/fold{str(count)}_val.csv'

    # Carrega os dados
    df_general_train = load_data(train_path)
    df_general_test = load_data(test_path)

    # Retorna dataframes com as informações adquiridas
    return df_general_train, df_general_test

def main():

    # Carrega e processa os dados
    process_data()

main()

