import pandas as pd
import keras
import tensorflow
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense
from tensorflow.python.keras.layers import Reshape
from scipy.stats import zscore
import os
from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.metrics import Accuracy

first_folder_session = "C:/Users/carmi/Desktop/Biometria/SEED_IV Database/SEED_IV Database/eeg_feature_smooth/1/"
df = pd.read_excel("C:/Users/carmi/Desktop/Biometria/SEED_IV Database/SEED_IV Database/SEED-IV_stimulation.xlsx",sheet_name="four emotions - experiment 1")
l = [x for x in df["Label"].values]
l = l[:len (l)-2]
l = [int (x) for x in l]
# Itera su tutti i file nella cartella
n = 0
for file in os.listdir(first_folder_session):
    if file.endswith(".mat"):  # Verifica se il file ha estensione .mat
        percorso_file = os.path.join(first_folder_session, file)  # Percorso completo del file .mat
        first_session = loadmat(percorso_file)  # Carica il file .mat
        key = list(first_session.keys())[3:]
        n_key = len(key)
        label_cont = 0
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
            np.save("C:/Users/carmi/Desktop/Biometria/SEED_IV Database/SEED_IV Database/dati/" + str(l[label_cont]) + "/matrice_" + str(n), x)
            n+=1
            label_cont += 1


first_folder_session = "C:/Users/carmi/Desktop/Biometria/SEED_IV Database/SEED_IV Database/eeg_feature_smooth/2/"
df = pd.read_excel("C:/Users/carmi/Desktop/Biometria/SEED_IV Database/SEED_IV Database/SEED-IV_stimulation.xlsx",sheet_name="four emotions - experiment 2")
l = [x for x in df["Label"].values]

l = [int (x) for x in l]

# Itera su tutti i file nella cartella
for file in os.listdir(first_folder_session):
    if file.endswith(".mat"):  # Verifica se il file ha estensione .mat
        percorso_file = os.path.join(first_folder_session, file)  # Percorso completo del file .mat
        first_session = loadmat(percorso_file)  # Carica il file .mat
        key = list(first_session.keys())[3:]
        n_key = len(key)
        label_cont = 0
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
            np.save("C:/Users/carmi/Desktop/Biometria/SEED_IV Database/SEED_IV Database/dati/" + str(l[label_cont]) + "/matrice_" + str(n), x)
            n+=1
            label_cont += 1


# Creazione di una lista per le matrici normalizzate
normalized_matrices = []
first_folder_session = "C:/Users/carmi/Desktop/Biometria/SEED_IV Database/SEED_IV Database/eeg_feature_smooth/3/"
df = pd.read_excel("C:/Users/carmi/Desktop/Biometria/SEED_IV Database/SEED_IV Database/SEED-IV_stimulation.xlsx",sheet_name="four emotions - experiment 3")
l = [x for x in df["Label"].values]
l = [int (x) for x in l]

for file in os.listdir(first_folder_session):
    if file.endswith(".mat"):  # Verifica se il file ha estensione .mat
        percorso_file = os.path.join(first_folder_session, file)  # Percorso completo del file .mat
        first_session = loadmat(percorso_file)  # Carica il file .mat
        key = list(first_session.keys())[3:]
        n_key = len(key)
        label_cont = 0
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
            np.save("C:/Users/carmi/Desktop/Biometria/SEED_IV Database/SEED_IV Database/dati/" + str(l[label_cont]) + "/matrice_" + str(n), x)
            n += 1
            label_cont += 1






'''

for matrix in x:
    normalized_matrix = zscore(matrix, axis=0)
    normalized_matrices.append(normalized_matrix)

normalized_batch_x = np.array(normalized_matrices)
'''