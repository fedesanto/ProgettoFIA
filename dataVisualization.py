import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)      # Mostra tutte le colonne quando stampi il dataframe
pd.set_option("display.width", None)            # Non tornare a capo quando stampi il dataframe

def showData(dataframe, save_plot=None):
    """
        Funzione per visualizzare un dataframe e ricavare informazioni su di esso. Saranno stampate a video, in successione:
            1) Informazioni generali (numero di colonne, etichette delle colonne, tipo dei dati contenuti nelle colonne,
                                      range degli indici, valori non nulli e memoria utlizzata)
            2) Le prime 10 righe
            3) Il numero di righe con almeno un valore nullo
            4) Il numero di righe identiche
            5) Il numero di righe con descrizioni non significative (meno di 40 caratteri)
            6) Il numero totale di categorie differenti
            7) Le 20 categorie più presenti

        Parametri:
            - dataframe, dataframe da analizzare
            - save_plot, se questo parametro viene impostato verrà memorizzato, all 'interno della directory "/Plots",
                         un grafico sotto forma di .png che mostra il numero di libri delle categorie più frequenti nel dataframe.
                         Il file creato potrà essere identificato come "save_plot".png
    """

    print("Informazioni generali sul dataframe:\n")
    print(dataframe.info())

    to_show = 10
    print(f"\nPrime {to_show} righe:\n")
    print(dataframe.head(to_show))

    len_threshold = 40
    null_rows = len(dataframe[ dataframe.isna().any(axis=1) ])
    identical_rows = dataframe.duplicated().sum()
    dummy_descriptions = len(dataframe[dataframe["Description"].str.len() < len_threshold])
    print(f"\nNumero di righe con almeno un valore nullo: {null_rows}")
    print(f"Numero righe identiche: {identical_rows}")
    print(f"Numero righe con descrizioni non significative (meno di {len_threshold} caratteri): {dummy_descriptions}")

    top_categories = 20
    categories = dataframe["Category"].str.split(r"\s*,\s*", expand=True).stack().str.strip()
    category_counts = categories.value_counts()
    top_categories_counts = category_counts.head(top_categories)


    print(f"\nNumero totale di differenti categorie: {len(category_counts)}")
    print(f"\nLe {top_categories} categorie più presenti:")
    print(top_categories_counts)

    if save_plot:
        plt.figure(figsize=(18, 4))
        plt.barh(top_categories_counts.index[::-1], top_categories_counts.values[::-1])
        plt.title(f"Conteggio {top_categories} categorie più frequenti")
        plt.ylabel("Categorie")
        plt.xlabel("Numero di libri")
        plt.savefig(f"Plots/{save_plot}.png")
        print(f"Memorizzato il plot delle categorie in: /Plots/{save_plot}.png")