from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import csv
from sklearn import preprocessing
from rna import ArtificialNeuralNetwork

filtered = []
with open("data/breast-cancer-wisconsin.csv", "r") as csvfile:
    # Seu código aqui
    csvr = csv.reader(
        filter(lambda l: l[6] != "?", csvfile),
        delimiter=",",
    )

    for l in csvr:
        if "?" not in l[6]:
            filtered.append(l)
        pass

np_array = np.array(filtered, dtype='int64')
np_normalized = preprocessing.MinMaxScaler().fit_transform(np_array)

X = np_normalized[:,1:10]
Y = np_normalized[:,10]

seed = 7
test_size = 0.3

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size, random_state=seed)

print(X_train)
print("======")
print(X_test)
print("======")
print(Y_train)
print("======")
print(Y_test)

neural_net = ArtificialNeuralNetwork(input_dim=9)
neural_net.train(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_test, Y_test))

# Avaliar o modelo
accuracy, conf_matrix, roc_auc = neural_net.evaluate(X_test, Y_test)
print(f"Accuracy: {accuracy}")
print(f'Acurácia no conjunto de teste: {accuracy * 100:.2f}%')
print("Confusion Matrix:")
print(conf_matrix)
print(f"ROC AUC: {roc_auc}")
print(f'Área sob a Curva ROC: {roc_auc:.2f}')

# Plotar gráficos
neural_net.plot_loss()
neural_net.plot_roc_curve(X_test, Y_test)
