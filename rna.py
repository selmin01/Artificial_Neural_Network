import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc


class ArtificialNeuralNetwork:
    def __init__(self, input_dim):
        self.model = Sequential()
        self.model.add(Dense(10, input_dim=input_dim, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        # Defina o otimizador com uma taxa de aprendizado específica
        self.custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(loss='binary_crossentropy', optimizer=self.custom_optimizer, metrics=['accuracy'])

    def train(self, X_train, Y_train, epochs=50, batch_size=32, validation_data=None):
        self.history = self.model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

    def evaluate(self, X_test, Y_test):
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.round(y_pred)
        accuracy = accuracy_score(Y_test, y_pred_classes)
        conf_matrix = confusion_matrix(Y_test, y_pred_classes)
        fpr, tpr, _ = roc_curve(Y_test, y_pred)
        roc_auc = auc(fpr, tpr)

        return accuracy, conf_matrix, roc_auc, y_pred_classes

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
        os.makedirs('./imgs', exist_ok=True)
        # Salve a imagem
        plt.savefig(os.path.join('./imgs', 'result_loss.png'))
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
        plt.savefig('./imgs/result_roc_curve.png')
        plt.show()

    def plot_confusion_matrix(self, Y_test, y_pred_classes):
        conf_matrix = confusion_matrix(Y_test, y_pred_classes)

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Negative", "Positive"],
                    yticklabels=["Negative", "Positive"])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig('./imgs/result_confusion_matrix.png')
        plt.show()

    def plot_test_metrics(self):
        # test_accuracy = self.history.history['accuracy']
        # test_loss = self.history.history['val_loss']

        # epochs = range(1, len(test_accuracy) + 1)

        # plt.figure(figsize=(12, 5))

        # # Plotar a precisão do teste
        # plt.subplot(1, 2, 1)
        # plt.plot(epochs, test_accuracy, 'b', label='Test Accuracy')
        # plt.title('Test Accuracy')
        # plt.xlabel('Epochs')
        # plt.ylabel('Accuracy')
        # plt.legend()

        # # Plotar a perda do teste
        # plt.subplot(1, 2, 2)
        # plt.plot(epochs, test_loss, 'r', label='Test Loss')
        # plt.title('Test Loss')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.legend()

        train_loss = self.history.history['loss']
        test_loss = self.history.history['val_loss']
        train_accuracy = self.history.history['accuracy']
        test_accuracy = self.history.history['val_accuracy']

        epochs = range(1, len(train_loss) + 1)

        # Plotar Loss
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss, label='Training Loss')
        plt.plot(epochs, test_loss, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss over Epochs')
        plt.legend()

        # Plotar Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracy, label='Training Accuracy')
        plt.plot(epochs, test_accuracy, label='Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Test Accuracy over Epochs')
        plt.legend()

        plt.tight_layout()
        plt.savefig('./imgs/result_test_metrics.png')
        plt.show()

# Exemplo de uso:
# neural_net = ArtificialNeuralNetwork(input_dim=9)
# neural_net.train(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_test, Y_test))
# accuracy, conf_matrix, roc_auc = neural_net.evaluate(X_test, Y_test)
# neural_net.plot_loss()
# neural_net.plot_roc_curve(X_test, Y_test)
# neural_net.plot_confusion_matrix(Y_test, y_pred_classes)
# neural_net.plot_test_metrics()
