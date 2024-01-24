import pandas as pd

from os import listdir   # Utile per recuperare i nomi dei file dei modelli addestrati
from joblib import load  # Permette di recuperare un oggetto serializzato in un file
from dataPreparation import preprocessAuthors, preprocessDescription   # Necessari per processari i dati inseriti dall'utente

# Recupero i nomi dei modelli precedentemente addestrati e serializzati
modelNames = [modelName.replace(".joblib", "") for modelName in listdir("Models/Classificators") if ".joblib" in modelName]

if len(modelNames) == 0:        # Se non ci sono modelli addestrati, arresto il programma
    exit("Non sono disponibili modelli di classificazione addestrati")


print("Scegliere tra uno dei seguenti modelli addestrati:")
for modelName in modelNames:
    print(modelName)

while True:     # Ciclo di richiesta input del giusto nome del modello da utilizzare
    selection = input("\nNome del modello scelto: ")
    if selection in modelNames:
        break
    else:
        print("Il nome indicato non è presente nell'elenco riportato")


while True:  # Ciclo di richiesta input del numero di libri che si vuole predirre
    try:
        rows = int(input("\nIndicare il numero di libri che si intende inserire: "))

        if rows <= 0:
            print("Indicare un numero intero superiore o uguale a 1")
        else:
            break
    except ValueError:
        print("Indicare un numero intero superiore o uguale a 1")

titles = []
descriptions = []
authors = []
for i in range(rows):   # Richiedo titolo, descrizione e autori per ciascun libro
    print(f"\nLibro {i+1}")
    titles.append(input("Titolo: "))
    descriptions.append(input("Descrizione: "))
    authors.append(input("Autori (separati da virgola): "))


print("\nCalcolo delle predizioni...")
toPredict = pd.DataFrame(data = {"Title" : titles, "Description": descriptions, "Authors": authors})

with open('stopwords/description_stopwords.txt', 'r') as fd:  # Recupero le stopwords per le descrizioni
    description_stopwords = [word.replace("\n", "") for word in fd]

with open('stopwords/authors_stopwords.txt', 'r') as fd:      # Recupero le stopwords per gli autori
    authors_stopwords = [word.replace("\n", "") for word in fd]

preprocessDescription(toPredict, description_stopwords)    # Processo descrizioni e autori
preprocessAuthors(toPredict, authors_stopwords)

model = load(f"Models/Classificators/{selection}.joblib")  # Recupero il modello serializzato indicato
predictions = model.predict(toPredict)  # Calcolo le predizioni

print("\nPredizioni:")  # Stampo le predizioni
for index, prediction in enumerate(predictions):
    print(f"{titles[index]} -> {prediction}")