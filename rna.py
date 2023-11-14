import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc


class ArtificialNeuralNetwork:
    def __init__(self, input_dim):
        self.model = Sequential()
        self.model.add(Dense(10, input_dim=input_dim, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, X_train, Y_train, epochs=50, batch_size=32, validation_data=None):
        self.history = self.model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

    def evaluate(self, X_test, Y_test):
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.round(y_pred)
        accuracy = accuracy_score(Y_test, y_pred_classes)
        conf_matrix = confusion_matrix(Y_test, y_pred_classes)
        fpr, tpr, _ = roc_curve(Y_test, y_pred)
        roc_auc = auc(fpr, tpr)

        return accuracy, conf_matrix, roc_auc

    def plot_loss(self):
        train_loss = self.history.history['loss']
        test_loss = self.history.history['val_loss']

        plt.figure()
        plt.plot(train_loss, label='Training Loss')
        plt.plot(test_loss, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss over Epochs')
        plt.legend()
        plt.show()

    def plot_roc_curve(self, X_test, Y_test):
        y_pred = self.model.predict(X_test)
        fpr, tpr, _ = roc_curve(Y_test, y_pred)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

# Exemplo de uso:
# neural_net = ArtificialNeuralNetwork(input_dim=9)
# neural_net.train(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_test, Y_test))
# accuracy, conf_matrix, roc_auc = neural_net.evaluate(X_test, Y_test)
# neural_net.plot_loss()
# neural_net.plot_roc_curve(X_test, Y_test)
