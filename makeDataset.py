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
import random

#METODO 1
first_folder_session = "C:/Users/carmi/Desktop/Biometria/SEED_IV Database/SEED_IV Database/eeg_feature_smooth/1/"
df = pd.read_excel("C:/Users/carmi/Desktop/Biometria/SEED_IV Database/SEED_IV Database/SEED-IV_stimulation.xlsx",sheet_name="four emotions - experiment 1")
l = [x for x in df["Label"].values]
l = l[:len(l)-2]
l = [int(x) for x in l]
# Itera su tutti i file nella cartella
n = 0
num_soggetti=0
train_set=[]
test_set=[]
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
            #np.save("C:/Users/carmi/Desktop/Biometria/SEED_IV Database/SEED_IV Database/dati/" + str(l[label_cont]) + "/matrice_" + str(n), x)
            n += 1
            label_cont += 1
        random.shuffle(x)
        train_data = x[:16]
        train_set.append(train_data)
        test_data = x[:8]
        test_set.append(test_data)
train_set = np.array(train_set)
test_set = np.array(test_set)
print(train_set.shape)
print(test_set.shape)

'''
#METODO 2
train_set=[]
test_set=[]
n=0
num_soggetti=0
indice_soggetti=0
soggetti=[]
first_folder_session = "C:/Users/carmi/Desktop/Biometria/SEED_IV Database/SEED_IV Database/eeg_feature_smooth/1/"
df = pd.read_excel("C:/Users/carmi/Desktop/Biometria/SEED_IV Database/SEED_IV Database/SEED-IV_stimulation.xlsx",sheet_name="four emotions - experiment 1")
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
            soggetti.append(x)
            #np.save("C:/Users/carmi/Desktop/Biometria/SEED_IV Database/SEED_IV Database/dati/" + str(l[label_cont]) + "/matrice_" + str(n), x)
            n+=1
            label_cont += 1

    num_soggetti += 1
print(num_soggetti)
soggetti= np.array(soggetti)
random.shuffle(soggetti)
train_data = soggetti[:14]
train_set.append(train_data)
test_data = soggetti[14]
test_set.append(test_data)
train_set = np.array(train_set)
test_set = np.array(test_set)
print(train_set.shape)
print(test_set.shape)
'''
'''
#Metodo 3
first_folder_session = "C:/Users/carmi/Desktop/Biometria/SEED_IV Database/SEED_IV Database/eeg_feature_smooth/2/"
df = pd.read_excel("C:/Users/carmi/Desktop/Biometria/SEED_IV Database/SEED_IV Database/SEED-IV_stimulation.xlsx", sheet_name="four emotions - experiment 1")
l = [x for x in df["Label"].values]
l = l[:len(l)-2]
l = [int(x) for x in l]
n=0
label=[]
soggetti=[]
x_train_set_final = []
x_test_set_final = []
y_train_set_final = []
y_test_set_final = []
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
            soggetti.append(x)
            label.append(l)
            #np.save("C:/Users/carmi/Desktop/Biometria/SEED_IV Database/SEED_IV Database/dati/" + str(l[label_cont]) + "/matrice_" + str(n), x)
            n += 1
            label_cont += 1
        x_train_set, x_test_set, y_train_set, y_test_set = train_test_split(soggetti, label, test_size=0.2, random_state=69)
        x_train_set_final.append(x_train_set)
        x_test_set_final.append(x_test_set)
        y_train_set_final.append(y_train_set)
        y_test_set_final.append(y_test_set)

print(len(x_train_set_final))
'''


'''
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
            n += 1
            label_cont += 1


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
