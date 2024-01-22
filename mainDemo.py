import pandas as pd

from os import listdir
from re import sub
from joblib import load

modelNames = listdir("Models")  # Recupero i nomi dei modelli precedentemente addestrati e serializzati
modelNames = [sub(r"\.joblib", "", modelName) for modelName in modelNames]

if len(modelNames) == 0:
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
            print("Indicare un valore superiore o uguale a 1")
        else:
            break
    except ValueError:
        print("Indicare un numero intero superiore o uguale a 1")

titles = []
descriptions = []
authors = []
for i in range(rows):
    print(f"\nLibro {i+1}")
    titles.append(input("Titolo: "))
    descriptions.append(input("Descrizione: "))
    authors.append(input("Autori (separati da virgola): "))

print("\nCalcolo delle predizioni...")
model = load(f"Models/{selection}.joblib")  # Recupero il modello serializzato indicato

toPredict = pd.DataFrame(data = {"Title" : titles, "Description": descriptions, "Authors": authors})

predictions = model.predict(toPredict)

print("\nPredizioni:")
for index, prediction in enumerate(predictions):
    print(f"{titles[index]} -> {prediction}")