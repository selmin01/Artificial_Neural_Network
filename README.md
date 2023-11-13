# Artificial_Neural_Network


Projeto de Implementação e Treinamento de MLP para Classificação de Dados

Este projeto tem como objetivo explorar a implementação e treinamento de uma Rede Neural do tipo Multilayer Perceptron (MLP) para a classificação de dados. Utilizaremos as bibliotecas TensorFlow e scikit-learn para desenvolver e avaliar nosso modelo.
Conjunto de Dados

O conjunto de dados escolhido para este trabalho é o Breast Cancer Wisconsin (Original), disponível em UCI Machine Learning Repository. É importante destacar que o conjunto de dados possui valores ausentes, sendo necessária a realização da limpeza desses dados.
Passos para Realização do Trabalho
1. Carregar o Dataset

Inicialmente, carregaremos o conjunto de dados e realizaremos a limpeza dos dados faltantes.
2. Normalização dos Dados

Após a carga dos dados, aplicaremos a normalização para garantir que todas as características estejam na mesma escala. Utilizaremos o StandardScaler do scikit-learn para essa tarefa.
3. Dividir os Dados em Conjuntos de Treinamento e Teste (Método Holdout)

Usaremos o método holdout para dividir os dados em conjuntos de treinamento e teste. Esta divisão é crucial para avaliar o desempenho do modelo em dados não vistos.
4. Definir a Arquitetura da Rede Neural com TensorFlow

A arquitetura da MLP será definida utilizando a biblioteca TensorFlow. Ajustaremos o número de camadas ocultas, o número de neurônios em cada camada e as funções de ativação para otimizar a acurácia do modelo.
5. Definir o Otimizador (Algoritmo de Treino) - Adam

Utilizaremos o algoritmo de otimização Adam para calcular os pesos da rede neural. O Adam trabalha em conjunto com o backpropagation, que é crucial no treinamento de redes neurais.
6. Treinar o Modelo

Treinaremos o modelo usando o conjunto de treinamento e monitoraremos os valores de perda ao longo do processo de treinamento.
7. Avaliar o Modelo no Conjunto de Teste

Avaliaremos o desempenho do modelo no conjunto de teste, extraindo valores de perda para análise.
8. Avaliação e Comparação das Arquiteturas

Realizaremos uma avaliação comparativa das arquiteturas de MLP testadas, considerando:
a. Valores de acurácia
b. Assertividade (curva ROC, matriz confusão)
c. Gráficos de perda (treinamento e teste) para detectar underfitting ou overfitting.
Adicional: Variação da Taxa de Aprendizado

Para a melhor configuração encontrada, variaremos a taxa de aprendizado e repetiremos a avaliação do modelo.
Relatório Técnico

Ao final dos testes, será elaborado um relatório técnico que explicará as decisões tomadas, apresentando os valores e arquiteturas utilizados, juntamente com gráficos comparativos entre os modelos. O relatório incluirá insights sobre a influência da taxa de aprendizado nas métricas de avaliação.
