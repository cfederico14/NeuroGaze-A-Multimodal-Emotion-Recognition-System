import os
import random
import keras.optimizers
import numpy as np
import tensorflow as tf
from keras import Sequential, optimizers, losses
from keras.utils import Sequence
from sklearn.model_selection import StratifiedKFold
import shutil
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


class NumpyDataLoader(Sequence):
    def __init__(self, data_dir,  batch_size=32, preprocess=True, mean=0, std=1):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.class_list = sorted(os.listdir(data_dir))
        self.data = self.load_data()
        self.prep = preprocess
        self.split = StratifiedKFold(n_splits=len(self), shuffle=True)
        self.mean= np.expand_dims(mean, axis=(0, 1, 2))
        self.std = np.expand_dims(std, axis=(0, 1, 2))

    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, index):
        batch_data = [self.data[i] for i in index]
        random.shuffle(batch_data)
        x = []
        y = []
        for file_path, label in batch_data:
            data = np.load(file_path)
            x.append(data)
            y.append(label)

        x = np.array(x)
        y = np.array(y)
        if self.prep:
                x = (x-self.mean)/self.std
        return x, y

    def __iter__(self):
        self.lbl = [l for _, l in self.data]
        for _, batch_index in self.split.split(self.lbl, self.lbl):
            yield self[batch_index]

    def load_data(self):
        data = []
        for i, class_name in enumerate(self.class_list):
            class_dir = os.path.join(self.data_dir, class_name)
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                data.append((file_path, i))
                x = np.load(file_path)

        return data


source = "C:/Users/carmi/Desktop/Biometria/SEED_IV Database/SEED_IV Database/dati/"
dest = "C:/Users/carmi/Desktop/Biometria/SEED_IV Database/SEED_IV Database/splittati/"

for file in os.listdir(source):
    shutil.copytree(source + file, dest + "train/" + file + "/")
os.makedirs(dest + "test")

for dir in os.listdir(dest + "train"):
    files = os.listdir(dest + "train/" + dir)
    to_move = random.sample(files, round(len(files) * 0.2))
    os.makedirs(dest + "test/" + dir)
    for file in to_move:
        shutil.move(dest + "train/" + dir + "/" + file, dest + "test/" + dir + "/" + file)
mean = np.array([13.12549127, 11.93335623, 11.43988904, 10.71254144, 9.9640656,
                     13.12549127, 11.93335623, 11.43988904, 10.71254144, 9.9640656,
                     13.12549127, 11.93335623, 11.43988904, 10.71254144, 9.9640656,
                     13.12549127, 11.93335623, 11.43988904, 10.71254144, 9.9640656])
std = np.array([12.15270327, 11.04254002, 10.57943663, 9.92019554, 9.26625739,
                    12.15270327, 11.04254002, 10.57943663, 9.92019554, 9.26625739,
                    12.15270327, 11.04254002, 10.57943663, 9.92019554, 9.26625739,
                    12.15270327, 11.04254002, 10.57943663, 9.92019554, 9.26625739])

train_loader = NumpyDataLoader(data_dir="C:/Users/carmi/Desktop/Biometria/SEED_IV Database/SEED_IV Database/splittati/train/", batch_size=128, preprocess=True, mean=mean, std=std)
test_loader = NumpyDataLoader(data_dir="C:/Users/carmi/Desktop/Biometria/SEED_IV Database/SEED_IV Database/splittati/test/", batch_size=64, preprocess=True, mean=mean, std=std)
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(62, 64, 20)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(len(train_loader.class_list), activation='softmax'))

optimizer = keras.optimizers.Adam(learning_rate=0.001)

#Definizione della funzione di costo Cross Entropy

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

#Definizione della funzione per calcolare l'accuratezza


def calculate_accuracy(labels, logits):
    predictions = tf.argmax(logits, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), dtype=tf.float32))
    return accuracy


def train(model, optimizer, loss_fn, train_loader, epochs):
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        for batch_data, batch_labels in train_loader:
            with tf.GradientTape() as tape:
                # Calcola le predizioni del modello
                logits = model(batch_data)
                # Calcola il valore della funzione di costo
                loss_value = loss_fn(batch_labels, logits)

            # Calcola il gradiente dei parametri del modello rispetto alla funzione di costo
            gradients = tape.gradient(loss_value, model.trainable_variables)

            # Applica gli aggiornamenti dei parametri utilizzando l'ottimizzatore
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Calcola l'accuratezza del batch
            batch_accuracy = calculate_accuracy(batch_labels, logits)

            epoch_loss += loss_value.numpy()
            epoch_accuracy += batch_accuracy

        # Calcola la media delle metriche di loss e accuratezza per l'epoca corrente
        epoch_loss /= len(train_loader)
        epoch_accuracy /= len(train_loader)

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.4f}")


train(model, optimizer, loss_fn, train_loader, 10)


def test(model, loss_fn, test_loader):
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        for batch_data, batch_labels in test_loader:
            logits = model(batch_data)
            # Calcola il valore della funzione di costo
            loss_value = loss_fn(batch_labels, logits)
            # Calcola l'accuratezza del batch
            batch_accuracy = calculate_accuracy(batch_labels, logits)

            epoch_loss += loss_value.numpy()
            epoch_accuracy += batch_accuracy

        # Calcola la media delle metriche di loss e accuratezza per l'epoca corrente
        epoch_loss /= len(test_loader)
        epoch_accuracy /= len(test_loader)

        print("Test :" + f"Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.4f}")


test(model, loss_fn, test_loader)


