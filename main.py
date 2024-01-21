import pandas as pd
import numpy as np
import nltk
nltk.data.path.append("nltk_data")  # Necessario per indicare alla libreria "nltk" dove si trovano i dati di cui ha bisogno

from dataVisualization import *
from dataPreparation import *
from dataModelling import  *

from sklearn.model_selection import train_test_split


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
description_stopwords = stopwords.words('english') + ["life", "book", "one", "new", "time", "world", "find", "using", "use"]
authors_stopwords = ["book", "books", "editor", "editors", "mr", "mrs", "dr", "jr", "magazine", "and"]

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
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=3, stratify=Y)

model_names = ["LinearSVC", "LogisticRegression", "SGDClassifier", "MultinomialNB", "ComplementNB"]
models = {}

print("Addestramento dei modelli di classificazione")
for model_name in model_names:
    print(f"\nAddestramento del modello {model_name}...")
    model, fitTime = trainClassificator(X_train, Y_train, model=model_name, returnFitTime=True)
    print(f"Addestramento concluso in {round(fitTime, 2)} secondi")
    print("\nTesting del modello...\n")
    predictionTime = testClassificator(X_test, Y_test, model, returnPredictionTime=True,
                                       saveConfusionMatrix=f"{model_name}_confusion_matrix")
    print(f"\nPredizione effettuata in {round(predictionTime, 2)} secondi")

    models[model_name] = model


bestScore = 0
best_model = None
for model_name, model in models.items():
    score = model.score(X_test, Y_test)
    if score > bestScore:
        bestScore = score
        best_model = model_name

print(f"\nSecondo l'accuracy, il migliore modello è {best_model}, con uno score di {bestScore}")
print("---------------------------------------------")





print("\n---------------------------------------------")

print("Addestrando il modello KMeans...")
model,time = trainClusters(df[["Description","Authors"]],"KMeans")
print(f"Addestramento completato, tempo impiegato: {time}s")
clusterAnalysis(df[["Description","Authors"]], model)

print("Addestrando il modello MiniBatchKMeans...")
model,time = trainClusters(df[["Description","Authors"]],"MiniBatchKMeans")
print(f"Addestramento completato, tempo impiegato: {time}s")
clusterAnalysis(df[["Description","Authors"]], model)

print("Addestrando il modello SpectralClustering...")
model,time = trainClusters(df[["Description","Authors"]],"SpectralClustering")
print(f"Addestramento completato, tempo impiegato: {time}s")
clusterAnalysis(df[["Description","Authors"]], model)

print("\n---------------------------------------------")