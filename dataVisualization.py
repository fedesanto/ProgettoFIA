import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from wordcloud import WordCloud

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
            8) Se save_plot è diverso da 'None', viene creato un plot riportate il numero di libri per ciascuna delle 20 categorie più presenti

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


def createDescriptionWordCloud(dataframe, save_file = "description_wordcloud"):
    """
        Funzione che permette la creazione e salvataggio di un'immagine wordcloud per le descrizioni.
        Più nello specifico, vengono creati tanti wordcloud quante sono le categorie e ognuno di essi
        conterrà solo le descrizioni di libri appartenenti ad una specifica categoria

        Parametri:
            - dataframe, dataframe su cui effettuare l'operazione
            - save_file, nome del file (senza estensione finale) in cui salvare l'immagine wordcloud
    """
    n_categories = len(dataframe["Category"].unique())
    n_rows, n_cols = (int(n_categories/2), 2)

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(10, n_rows*3))
    fig.suptitle('WordCloud delle descrizioni', fontsize=40, y=1)
    fig.tight_layout()

    contAx = 0
    for category in dataframe["Category"].unique():
        descriptions = dataframe[ dataframe["Category"] == category ]["Description"].tolist()
        text = ' '.join(descriptions)
        wordcloud = WordCloud(background_color='white').generate(text)

        ax[int(contAx/2), contAx%2].imshow(wordcloud)
        ax[int(contAx/2), contAx%2].axis('off')
        ax[int(contAx/2), contAx%2].set_title(category, fontsize=20, pad=10)

        contAx += 1

    plt.savefig(f"Plots/{save_file}.png")


def createAuthorsWordCloud(dataframe, save_file = "authors_wordcloud"):
    """
        Funzione che permette la creazione e salvataggio di un'immagine wordcloud per gli autori.
        Più nello specifico, vengono creati tanti wordcloud quante sono le categorie e ognuno di essi
        conterrà solo gli autori di libri appartenenti ad una specifica categoria

        Parametri:
            - dataframe, dataframe su cui effettuare l'operazione
            - save_file, nome del file (senza estensione finale) in cui salvare l'immagine wordcloud
    """
    n_categories = len(dataframe["Category"].unique())
    n_rows, n_cols = (int(n_categories/2), 2)

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(10, n_rows*3))
    fig.suptitle('WordCloud degli autori', fontsize=40, y=1)
    fig.tight_layout()

    contAx = 0
    for category in dataframe["Category"].unique():
        authors = dataframe[ dataframe["Category"] == category ]["Authors"].tolist()
        text = ' '.join(authors)
        wordcloud = WordCloud(background_color='white').generate(text)

        ax[int(contAx/2), contAx%2].imshow(wordcloud)
        ax[int(contAx/2), contAx%2].axis('off')
        ax[int(contAx/2), contAx%2].set_title(category, fontsize=20, pad=10)

        contAx += 1

    plt.savefig(f"Plots/{save_file}.png")