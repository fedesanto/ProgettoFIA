import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from dataVisualization import *
from dataPreparation import *
from dataModelling import *

pd.set_option("display.max_columns", None)      # Mostra tutte le colonne quando stampi il dataframe
pd.set_option("display.width", None)            # Non tornare a capo quando stampi il dataframe

if not os.path.isdir("Plots"):      # Mi assicuro che la cartella "Plots" esista
    os.mkdir("Plots")

if not os.path.isdir("Models"):  # Mi assicuro che la cartella "Models" esista
     os.mkdir("Models")


df = pd.read_csv("Data/BooksDataset.csv", usecols=["Title", "Description", "Authors", "Category"])

# Mostro alcune informazioni del dataset iniziale
showData(df, save_plot="dati_iniziali")

# Ripulisco il dataset
print("\n---------------------------------------------")
print("Pulizia dataset")
cleanData(df)
print("---------------------------------------------")

# Rinomino categorie per accorparle in altre simili
print("\n---------------------------------------------")
replacementList = [
    ("Religious", "Religion"),
    ("Christian Life", "Religion"),
    ("Historical", "History"),
    ("Family", "Family & Relationships"),
    ("Pets", "Animal")
]
print("Rinominazione delle seguenti categorie:")
for replacement in replacementList:
    print(f"{replacement[0]} -> {replacement[1]}")

renameCategories(df, replacementList)
print("---------------------------------------------")

# Estraggo i libri delle sole categorie di cui sono interessato
print("\n---------------------------------------------")
extractList = ["Religion", "Romance", "Cooking", "History",
               "Business & Economics", "Thrillers", "Mystery & Detective",
               "Health & Fitness", "Art",  "Sports & Recreation", "Travel",
               "Fantasy", "Science", "Animals", "Computers", "House & Home"]
print("Estrazione dei libri appartenenti alle seguenti categorie:")
for category in extractList:
    print(category)

extractCategories(df, extractList)
print("---------------------------------------------")

# Bilancio il dataset eliminando righe di categorie troppo frequenti
print("\n---------------------------------------------")
print("Bilanciamento del dataset")
balanceCategories(df, 2800)
print("---------------------------------------------")

# Preprocesso i campi "Description" e "Authors" per renderli più puliti. Inoltre creo dei wordcloud prima e dopo avere processati i campi
print("\n---------------------------------------------")
with open('stopwords/description_stopwords.txt', 'r') as fd:  # Recupero le stopwords per le descrizioni
    description_stopwords = [word.replace("\n", "") for word in fd]

with open('stopwords/authors_stopwords.txt', 'r') as fd:      # Recupero le stopwords per gli autori
    authors_stopwords = [word.replace("\n", "") for word in fd]

print("Creo il wordcloud delle descrizioni non processate...")
createDescriptionWordCloud(df, save_file="wordcloud_descrizioni_nonprocessate")
print("Preprocessing delle descrizioni...")
preprocessDescription(df, description_stopwords)
print("Creo il wordcloud delle descrizioni processate...")
createDescriptionWordCloud(df, save_file="wordcloud_descrizioni_processate")

print("\nCreo il wordcloud degli autori non processati...")
createAuthorsWordCloud(df, save_file="wordcloud_autori_nonprocessati")
print("Preprocessing degli autori...")
preprocessAuthors(df, authors_stopwords)
print("Creo il wordcloud degli autori processati...")
createAuthorsWordCloud(df, save_file="wordcloud_autori_processati")

print()
cleanData(df, des_threshold = 20)
print("---------------------------------------------")

# Rimostro informazioni generali sul dataset dopo che è stato preparato
print("\n---------------------------------------------")
showData(df, save_plot="dati_postprocessati")
print("---------------------------------------------")


# Addestramento dei modelli di classificazione
print("\n---------------------------------------------")
X = df[["Description", "Authors"]]
Y = df["Category"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=3, stratify=Y)  # Suddivido i dati di addestramento da quelli di test

model_names = ["LinearSVC", "LogisticRegression", "SGDClassifier", "MultinomialNB", "ComplementNB"]  #Definisco i modelli che voglio addestrare
models = {}

print("Addestramento dei modelli di classificazione")
for model_name in model_names:      # Addesstro e valuto tutti i modelli specificati in "model_names"
    print(f"\nAddestramento del modello {model_name}...")
    model, fitTime = trainClassificator(X_train, Y_train, model=model_name, returnFitTime=True, saveEstimator=model_name)
    print(f"Addestramento concluso in {round(fitTime, 2)} secondi")

    print("\nTesting del modello...\n")
    predictionTime = testClassificator(X_test, Y_test, model, returnPredictionTime=True, saveConfusionMatrix=f"{model_name}_confusion_matrix")
    print(f"\nTesting concluso in {round(predictionTime, 2)} secondi")

    models[model_name] = model


bestScore = 0
best_model = None
for model_name, model in models.items():        # Determino qual è stato il miglior modello sulla base dell'accuracy score
    score = model.score(X_test, Y_test)
    if score > bestScore:
        bestScore = score
        best_model = model_name

print(f"\nSecondo l'accuracy, il migliore modello è {best_model}, con uno score di {round(bestScore, 3)}")

# Riaddestro il modello risultato migliore, identificandone la miglior configurazione di parametri per aumentarne ulteriormente il punteggio
# L'addestramento e la successiva valutazione avverranno su un set differenti di dati rispetto all'addestramento precedente

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=89, stratify=Y)

print(f"\nAddestramento di {best_model} ricercando la configurazione migliore...")
model, fitTime = trainClassificator(X_train, Y_train, model=best_model, returnFitTime=True, findBestEstimator=True, saveEstimator=f"{best_model}_best")
print(f"Addestramento concluso in {round(fitTime, 2)} secondi")

print("\nTesting del modello...\n")
predictionTime = testClassificator(X_test, Y_test, model, returnPredictionTime=True, saveConfusionMatrix=f"{best_model}_best_confusion_matrix")
print(f"\nTesting concluso in {round(predictionTime, 2)} secondi")
print("---------------------------------------------")


# Addestramento dei modelli di clustering
print("\n---------------------------------------------")

model_names = ["KMeans", "MiniBatchKMeans", "SpectralClustering"]

silhouette_models={}
models = {}
print("Addestramento dei modelli di clustering")
for model_name in model_names:
    print(f"\nAddestramento del modello {model_name}...")
    model, fitTime = trainClusters(df[["Description","Authors"]],model_name)
    print(f"Addestramento concluso in {round(fitTime, 2)} secondi")
    print("\nAnalisi del modello...\n")

    silhouette_models[model_name] = clusterAnalysis(df[["Description","Authors"]], model)
    models[model_name] = model

bestScore = 0
best_model = None
for model_name, model in models.items():        # Determino qual è stato il miglior modello sulla base del Silhouette score
    score = silhouette_models[model_name]
    if score > bestScore:
        bestScore = score
        best_model = model_name

print(f"\nSecondo il Silhouette score, il migliore modello è {best_model}, con uno score di {round(bestScore, 3)}")
# Riaddestro il modello risultato migliore, identificandone la miglior configurazione di parametri per aumentarne ulteriormente il punteggio
print(f"\nAddestramento di {best_model} ricercando la configurazione migliore...")
model, fitTime = trainClusters(df[["Description","Authors"]], model=best_model, findBestEstimator=True, saveEstimator=f"{best_model}_best")
print(f"Addestramento concluso in {round(fitTime, 2)} secondi")


print("\nAnalisi del modello...\n")
clusterAnalysis(df[["Description","Authors"]], model)
print("\n---------------------------------------------")