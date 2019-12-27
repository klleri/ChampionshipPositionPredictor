# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 20:16:18 2019

@author: Lucas
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import numpy as np
from sklearn.metrics import confusion_matrix

# Data preprocessing
# Pré processamento de dados
text = open("Data-Camp-Brasileiro_1.csv", "r")
text = ''.join([i for i in text]).replace(";", ",")
x = open("output.csv","w")
x.writelines(text)
x.close()
tabela = pd.read_csv('output.csv')
tabela = tabela.drop(['ANO', 'Time', 'Estados','Aproveitamento', 'Saldo Gols'],axis=1)

# Separating class from predictors  || Separando a classe dos previsores
previsores = tabela.iloc[:, 1:8].values
classe = tabela.iloc[:, 0].values

# Converting discrete numeric variables to dummy variables  || Convertendo as variáveis numericas discretas em variáveis ​​dummy
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)
# First 1000...     || Primeiro  1000...
# Second 0100...    || Segundo   0100...
# Third 0010...     || Terceiro  0010...

# Separating 75% from the base for training and 25% for testing || Separando 75% da base para treinamento e 25% para testar
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe_dummy, test_size=0.25)
# Construction of a network with 7 inputs, 3 hidden layers of 40 neurons   || Contrução de uma rede com 7 entradas, 3 camadas ocultas de 40 neuronios 
# Each one of the 20 softmax´s output that returns a probability for each output || Cada uma das 20 saídas da softmax que retorna uma probabilidade para cada saída
classificador = Sequential()
classificador.add(Dense(units = 50, activation = 'relu', input_dim = 6))
classificador.add(Dense(units = 50, activation = 'relu'))
classificador.add(Dense(units = 50, activation = 'relu'))
classificador.add(Dense(units = 20, activation = 'softmax'))

# optimizer = 'adam' é para ter uma melhoria na descida no gradiente estocástico
# loss = 'categorical_crossentropy' pois temos um problema que não é binario
# metrics = ['categorical_accuracy'] para fazer a avaliação do algoritmo 
classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['categorical_accuracy'])  # acuracia ponderada e acuracia geometrica
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 1,epochs = 1000)


# Using the answers to prepare a confusion matrix || Pegando as repostas para preparar uma matriz de confusão
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)
classe_teste2 = [np.argmax(t) for t in classe_teste]
previsoes2 = [np.argmax(t) for t in previsoes]
matriz = confusion_matrix(previsoes2, classe_teste2)


def jogos():
    global vitorias, empates, derrotas, golsPro, golsContra, pontos, predict
    vitorias = int(input("Number of victories: "))
    empates     = int(input("Number of draws: "))
    derrotas    = int(input("Number of Defeats: "))
    golsPro     = int(input("Goals for: "))
    golsContra  = int(input("Goals against: "))
    pontos      = ((vitorias * 3) + (empates * 1))
    predict     = np.array([[pontos, vitorias, empates, derrotas , golsPro, golsContra]])
    
jogos()

if((vitorias + empates + derrotas) == 38):
    previsao = classificador.predict(predict)
    i = 1
    for a in previsao[0]:
        if i < 10: 
            print('0{}º Place : {:.6f}'.format(i, a))
            i = i + 1
        else:
            print('{}º Place : {:.6f}'.format(i, a))
            i = i + 1
else:
    while (((vitorias + empates + derrotas) > 38) or ((vitorias + empates + derrotas) < 38)):
        print('Number of matches greater than or equal to 38 rounds')
        jogos()







