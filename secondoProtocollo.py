import pandas as pd
import keras
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from scipy.stats import zscore
import os
from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.metrics import Accuracy
import random

accuracy = []

mean = np.array([13.12549127, 11.93335623, 11.43988904, 10.71254144, 9.9640656,
                     13.12549127, 11.93335623, 11.43988904, 10.71254144, 9.9640656,
                     13.12549127, 11.93335623, 11.43988904, 10.71254144, 9.9640656,
                     13.12549127, 11.93335623, 11.43988904, 10.71254144, 9.9640656])
std = np.array([12.15270327, 11.04254002, 10.57943663, 9.92019554, 9.26625739,
                    12.15270327, 11.04254002, 10.57943663, 9.92019554, 9.26625739,
                    12.15270327, 11.04254002, 10.57943663, 9.92019554, 9.26625739,
                    12.15270327, 11.04254002, 10.57943663, 9.92019554, 9.26625739])
def make_model():
    model = Sequential()
    # Crea un livello convoluzionale 2D che esegue la convoluzione su "un'immagine" di input.
    # Utilizzando un kernel di dimensione 3x3 e una funzione di attivazione "relu" per introdurre non linearità nei dati di output.
    # l'immagine di input ha una dimensione di 62x64 pixel con 20 canali (canali di immagini o caratteristiche).
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(62, 64, 20)))
    # crea un livello di pooling 2D che riduce la dimensione spaziale dell'immagine di input campionando i valori massimi in regioni specifiche.
    # Utilizzando una finestra di pooling 2x2 per ridurre la dimensione dell'immagine di input.
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    model.add(
        Dense(128, activation='relu'))  # conversione di dati in un vettore 1D e da uno o più livelli fully connected
    model.add(Dense(4, activation='softmax'))

    return model


# Subject-Independent Experiment
first_folder_session = "C:/Users/carmi/Desktop/Biometria/SEED_IV Database/SEED_IV Database/eeg_feature_smooth/1/"
df = pd.read_excel("C:/Users/carmi/Desktop/Biometria/SEED_IV Database/SEED_IV Database/SEED-IV_stimulation.xlsx",sheet_name="four emotions - experiment 1")
l = [x for x in df["Label"].values]
l = l[:len(l) - 2]
l = [int(x) for x in l]
# Itera su tutti i file nella cartella
n = 0
dati = []
for file in os.listdir(first_folder_session):
    if file.endswith(".mat"):  # Verifica se il file ha estensione .mat
        percorso_file = os.path.join(first_folder_session, file)  # Percorso completo del file .mat
        first_session = loadmat(percorso_file)  # Carica il file .mat
        key = list(first_session.keys())[3:]
        n_key = len(key)
        label_cont = 0
        soggetti = []
        for i in range(0, n_key, 4):
            x1 = first_session[key[i]]
            padding = 64 - x1.shape[1]
            x1 = np.pad(x1, ((0, 0), (0, padding), (0, 0)), mode='constant', constant_values=0)
            x2 = first_session[key[i]]
            padding = 64 - x2.shape[1]
            x2 = np.pad(x2, ((0, 0), (0, padding), (0, 0)), mode='constant', constant_values=0)
            x3 = first_session[key[i]]
            padding = 64 - x3.shape[1]
            x3 = np.pad(x3, ((0, 0), (0, padding), (0, 0)), mode='constant', constant_values=0)
            x4 = first_session[key[i]]
            padding = 64 - x4.shape[1]
            x4 = np.pad(x4, ((0, 0), (0, padding), (0, 0)), mode='constant', constant_values=0)
            x = np.concatenate([x1, x2, x3, x4], 2)
            x = (x-mean)/std
            soggetti.append(x)
            n += 1
            label_cont += 1
        dati.append(soggetti)

for i in range(len(dati)):
    X_train = dati[:i]+dati[i+1:] if i<len(dati)-1 else dati[:i]
    y_train = l*len(X_train)
    X_test = dati[i]
    y_test = l
    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
    y_train_encoded = to_categorical(y_train)
    y_test_encoded = to_categorical(y_test)
    X_train = X_train.reshape(len(X_train) * 24, 62, 64, 20)

    model = make_model()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=1)
    # Addestramento del modello con EarlyStopping
    model.fit(X_train, y_train_encoded, epochs=1, batch_size=4,
              validation_data=(X_test, y_test_encoded), callbacks=[early_stopping])

    pred = model(X_test)
    predictions = tf.argmax(pred, axis=1)
    accuracy.append(tf.reduce_mean(tf.cast(tf.equal(y_test, predictions), dtype=tf.float32)))




first_folder_session = "C:/Users/carmi/Desktop/Biometria/SEED_IV Database/SEED_IV Database/eeg_feature_smooth/2/"
df = pd.read_excel("C:/Users/carmi/Desktop/Biometria/SEED_IV Database/SEED_IV Database/SEED-IV_stimulation.xlsx",sheet_name="four emotions - experiment 2")
l = [x for x in df["Label"].values]
l = [int(x) for x in l]
# Itera su tutti i file nella cartella
n = 0
dati = []
for file in os.listdir(first_folder_session):
    if file.endswith(".mat"):  # Verifica se il file ha estensione .mat
        percorso_file = os.path.join(first_folder_session, file)  # Percorso completo del file .mat
        first_session = loadmat(percorso_file)  # Carica il file .mat
        key = list(first_session.keys())[3:]
        n_key = len(key)
        label_cont = 0
        soggetti = []
        for i in range(0, n_key, 4):
            x1 = first_session[key[i]]
            padding = 64 - x1.shape[1]
            x1 = np.pad(x1, ((0, 0), (0, padding), (0, 0)), mode='constant', constant_values=0)
            x2 = first_session[key[i]]
            padding = 64 - x2.shape[1]
            x2 = np.pad(x2, ((0, 0), (0, padding), (0, 0)), mode='constant', constant_values=0)
            x3 = first_session[key[i]]
            padding = 64 - x3.shape[1]
            x3 = np.pad(x3, ((0, 0), (0, padding), (0, 0)), mode='constant', constant_values=0)
            x4 = first_session[key[i]]
            padding = 64 - x4.shape[1]
            x4 = np.pad(x4, ((0, 0), (0, padding), (0, 0)), mode='constant', constant_values=0)
            x = np.concatenate([x1, x2, x3, x4], 2)
            x = (x-mean)/std
            soggetti.append(x)
            n += 1
            label_cont += 1
        dati.append(soggetti)

for i in range(len(dati)):
    X_train = dati[:i]+dati[i+1:] if i<len(dati)-1 else dati[:i]
    y_train = l*len(X_train)
    X_test = dati[i]
    y_test = l
    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
    y_train_encoded = to_categorical(y_train)
    y_test_encoded = to_categorical(y_test)
    X_train = X_train.reshape(len(X_train) * 24, 62, 64, 20)

    model = make_model()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=1)
    # Addestramento del modello con EarlyStopping
    model.fit(X_train, y_train_encoded, epochs=1, batch_size=4,
              validation_data=(X_test, y_test_encoded), callbacks=[early_stopping])

    pred = model(X_test)
    predictions = tf.argmax(pred, axis=1)
    accuracy.append(tf.reduce_mean(tf.cast(tf.equal(y_test, predictions), dtype=tf.float32)))


first_folder_session = "C:/Users/carmi/Desktop/Biometria/SEED_IV Database/SEED_IV Database/eeg_feature_smooth/3/"
df = pd.read_excel("C:/Users/carmi/Desktop/Biometria/SEED_IV Database/SEED_IV Database/SEED-IV_stimulation.xlsx",sheet_name="four emotions - experiment 3")
l = [x for x in df["Label"].values]
l = [int(x) for x in l]
# Itera su tutti i file nella cartella
n = 0
dati = []
for file in os.listdir(first_folder_session):
    if file.endswith(".mat"):  # Verifica se il file ha estensione .mat
        percorso_file = os.path.join(first_folder_session, file)  # Percorso completo del file .mat
        first_session = loadmat(percorso_file)  # Carica il file .mat
        key = list(first_session.keys())[3:]
        n_key = len(key)
        label_cont = 0
        soggetti = []
        for i in range(0, n_key, 4):
            x1 = first_session[key[i]]
            padding = 64 - x1.shape[1]
            x1 = np.pad(x1, ((0, 0), (0, padding), (0, 0)), mode='constant', constant_values=0)
            x2 = first_session[key[i]]
            padding = 64 - x2.shape[1]
            x2 = np.pad(x2, ((0, 0), (0, padding), (0, 0)), mode='constant', constant_values=0)
            x3 = first_session[key[i]]
            padding = 64 - x3.shape[1]
            x3 = np.pad(x3, ((0, 0), (0, padding), (0, 0)), mode='constant', constant_values=0)
            x4 = first_session[key[i]]
            padding = 64 - x4.shape[1]
            x4 = np.pad(x4, ((0, 0), (0, padding), (0, 0)), mode='constant', constant_values=0)
            x = np.concatenate([x1, x2, x3, x4], 2)
            x = (x-mean)/std
            soggetti.append(x)
            n += 1
            label_cont += 1
        dati.append(soggetti)

for i in range(len(dati)):
    X_train = dati[:i]+dati[i+1:] if i<len(dati)-1 else dati[:i]
    y_train = l*len(X_train)
    X_test = dati[i]
    y_test = l
    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
    y_train_encoded = to_categorical(y_train)
    y_test_encoded = to_categorical(y_test)
    X_train = X_train.reshape(len(X_train) * 24, 62, 64, 20)


    model = make_model()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=1)
    # Addestramento del modello con EarlyStopping
    model.fit(X_train, y_train_encoded, epochs=1, batch_size=4,
              validation_data=(X_test, y_test_encoded), callbacks=[early_stopping])

    pred = model(X_test)
    predictions = tf.argmax(pred, axis=1)
    accuracy.append(tf.reduce_mean(tf.cast(tf.equal(y_test, predictions), dtype=tf.float32)))

print(len(accuracy))
print(sum(accuracy)/len(accuracy))





