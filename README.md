# Projeto de Implementação e Treinamento de MLP para Classificação de Dados

Neste trabalho, exploraremos a implementação e treinamento de uma Rede Neural do tipo Multilayer Perceptron (MLP) para a classificação de dados. Recomenda-se o uso das bibliotecas TensorFlow e scikit-learn.

## Conjunto de Dados
Utilizaremos o conjunto de dados Breast Cancer Wisconsin (Original), disponível em [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original). É importante notar que a base possui dados faltantes, sendo necessária a limpeza desses dados.

## Passos do Trabalho

### 1. Carregar o Dataset
   - Realizar o download do conjunto de dados e importá-lo para o ambiente de trabalho.

### 2. Normalização dos Dados
   - Aplicar a normalização dos dados para colocar todas as características em uma escala comum.

### 3. Dividir os Dados
   - Utilizar o método holdout para dividir os dados em conjuntos de treinamento e teste.

### 4. Definir a Arquitetura da Rede Neural
   - Utilizar TensorFlow para definir a arquitetura da MLP.

### 5. Definir um Otimizador (Adam)
   - Escolher o algoritmo de otimização Adam para calcular os pesos durante o treinamento.

### 6. Treinar o Modelo
   - Treinar a MLP usando os conjuntos de treinamento, extrair valores de perda durante o treinamento.

### 7. Avaliar o Modelo
   - Avaliar o desempenho do modelo no conjunto de teste, extrair valores de perda.

### 8. Avaliação e Comparação das Arquiteturas
   - a) Comparar valores de acuracidade.
   - b) Avaliar assertividade (curva ROC, matriz confusão).
   - c) Analisar gráficos de perda (treinamento e teste) para identificar underfitting ou overfitting.

### Ajustes na Arquitetura
   - Ajustar a quantidade de camadas ocultas, número de neurônios e funções de ativação para maximizar acurácia e assertividade (usar taxa de aprendizado de 0.01).

### Variação da Taxa de Aprendizado
   - Para a melhor configuração, variar a taxa de aprendizado e refazer a avaliação do modelo.

## Relatório Técnico
Ao final dos testes, elaborar um relatório técnico explicando as decisões tomadas, apresentando os valores e arquiteturas utilizados, e incluindo gráficos para comparar os modelos. Incluir análise de underfitting ou overfitting, além de sugestões para melhorias.

