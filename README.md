# Trabalho Final de Processamento de Imagens Médicas

## Enunciado

### 2  - Classificação de Imagens de tecido pulmonar (Tomografia Computadorizada)

A base de imagens lung_blocks.tar.gz no link abaixo possui imagens de tamanho 32x32 de imagens de seis classes de achados radiológicos de pulmão. Cada diretório da base é relativo a um paciente. O arquivo README tem informações sobre a nomenclatura das imagens. As imagens estão no formato DICOM. No trabalho você deve utilizar o Pyradiomics para extrair características dessas imagens.

#### Qual técnica de classificação deve ser usada?

O algoritmo do vizinho mais próximo deve ser utilizado. Você deverá dividir a base em 5-Folds e fazer o treino e teste para cada um desses folds. Tente fazer uma divisão equilibrada entre os Folds (quantidade de imagens e classes). Após a separação extrai-a um vetor de características (radiômicas) de todas as imagens e efetue os treinos/testes. Para saber a quantidade de imagens por classe por paciente utilize o script count.sh que está no link. Calcule as métricas de Sensibilidade, Especificidade e F1-Score.

[Link para o dataset](https://drive.google.com/drive/folders/1fpG-t1BX95AZvkjM-7CQgjRXFOOTrPSj?usp=sharing)

## O Trabalho

O artigo escrito pdoe ser encontrado neste reposítório sob o nome de [Artigo](), juntamente com os códigos utilizados para chegar neste resultado.

### O código

O código para separação do dataset em folds foi desenvolvido pelo aluno Felipe Duarte e o restante pela aluna Heloísa Viotto, tendo sido explicado e aprovado pelos demais integrantes da equipe.

A ordem de execução é: gerar_folds.py, gerar_mask.py, extract.py e classifier.py.

O código foi construido de fora modeluarizada, por isso cada seção as eguir irá explicar o funcionamento de cada um deles, em ordem de execução.

Os arquivos foram criados a partir de um único arquivo do Google Colaboratory, por isso algumas funções são parecidas e repetidas entre os arquivos.

#### Criar Folds


#### Criar Máscara

Este código foi desenvolvido para gerar as máscaras de cada imagem presente no dataset. 

A função 'generate_mask' possui o propósito de criar um arquivo de imagem 3D com o limiar de otsu e retornar o nome damáscara. Caso a máscara já tenha sido criada, é poupado o trabalho e retora o nome da máscara.

Já a função 'load_data' é responsável pelo carregamento dos dados de acordo com um arquivo csv passado por parÂmero em um dataframe o qual é retornado. 

A função 'process_data' define o nome do arquivo csv (criado pelo arquivo [gerar_folds](./gerar_folds.py)) de acordo com o número do fold passado por parâmetro. Ele carrega o arquivo e gera um dataframe que, neste arquivo, não será utilizado.

Por fim, a função 'main' apenas chama a process_data uma vez, o suficiente para criar todas as máscaras.


#### Extração de Características

As características extraídas são:

    * Estatísticas de Primeira Ordem (19 características);
    * Matriz de Co-ocorrência de Níveis de Cinza (GLCM) (24 características);
    * Matriz de Comprimento de Níveis de Cinza (GLRLM) (16 características);
    * Matriz de Comprimento de Execução de Níveis de Cinza (GLSZM) (16 características);
    * Matriz de Tamanho de Zona de Níveis de Cinza (NGTDM) (5 características);
    * Matriz de Diferença de Tons de Conza Vizinhos (NGTDM) (5 características);
    * Matriz de Dependência de Níveis de Cinza (GLDM) (14 características).

Dentro do [arquivo](./extract.py), é possível ver algumas funções para melhor organização do código. A primeira é a 'generate_mask' já utilizada no arquivo [gerar_mask](./gerar_mask.py).

A segunda função, 'extractor' é responsável por iniciar o extractor de características de acordo com as configurações presentes no arquivo [params](param.yaml).

A função 'characteristics' é responsável por extrair as características a partir de um dataframe passado por parâmetro. Algumas imagens deram erros durante a extração de característica, sendo eles armazenadas no arquivo [erros.txt](./erros.txt).

A função 'oad_data' também foi apresnetada em arquivos anteriores, juntamente com a 'process_data', no entanto a última apresneta algumas modificações, sendo elas a chamada da função de extração de características e a remoção de colunas com dados do pyradiomics.

Por fim, a função 'main' percorre cada uma dos arquivos CSV's de treino e teste e gera um novo arquivo para cada um deles contendo as características extraídas. Ou seja, as características são extraídas da mesma imagem 5 vezes. Apesar de não ser o método mais eficaz, a informação que deveria ser extraídas as características e salvá-las em um arquivo foi informada quando a base do projeto já estava pronta, sendo esta a solução menos custosa para o desenvolvimento. Na máquina Orval, este arquivo demora em torno de uma hora para ser executado.

#### Classificação das Imagens

O intuito desse trabalho foi de comparar os reusltados obtidos por três classificadores: o KNN, o Random Forest e o SVM.

Para encontrar os melhores parâmetros para cada uma dessas funções, foi testado algumas combinações de parâmetros e analisado o melhor resultado, sendo estes códigos e resultados presentes na pasta [grid_search](./grid_search/).

A primeira função trata-se da que calcula a matriz de confusão, sendo que é formado a matriz realxpredita e calculada as somas de uma matriz multiclase. O retorno é os valores de verdadeiro positivo (tp), verdadeiro negativo (tn), falso positivo (fp) e falso negativo(fn), respectivamente.

A função 'classifier'  trata-se juntamente do treino e da classificação de um algoritmo passado como argumento FUNCTION. O retorno são as métricas sensibilidade, especificidade e f1-score com 4 casas de precisão.

A função 'load_data' realiza o carregamento dos dados em CSV's com as características já extraídas e a separação dos dataframes com os dados e as labels, além da normalização dos dados.

Por fim, a função 'main' carrega os dados e os classificadores a cada fold, calculando as métricas e as armazenando no arquivo [results.txt](./results.txt).
