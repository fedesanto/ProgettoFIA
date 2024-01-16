import pandas as pd
import numpy as np

def cleanData(dataframe, des_threshold = 40):
    """
        Metodo per la pulizia del dataframe. In successione:
        1) Rimuove righe con almenon un campo nullo
        2) Rimuove righe duplicate
        3) Rimuove righe con descrizioni non significative (con meno di 'des_threshold' caratteri)

        Parametri:
        - dataframe, dataframe su cui effetture la pulizia
        - des_threshold, numero di caratteri minimo che una descrizione deve avere per considerarsi significativa (default = 40)
    """

    null_rows = len(dataframe[dataframe.isna().any(axis=1)])  # Numero di righe che hanno un qualsiasi campo nullo
    dataframe.dropna(inplace=True)   # Rimozione di righe con valori nulli
    print(f"Rimosse {null_rows} righe con valori nulli")

    identical_rows = dataframe.duplicated().sum()  # Numero di righe identiche
    dataframe.drop_duplicates(inplace=True)        # Rimozione di righe identiche
    print(f"Rimosse {identical_rows} righe duplicate")

    drop_indexes = dataframe[dataframe["Description"].str.len() < des_threshold].index   # Indici delle righe descrizioni da meno di 'des_threshold' caratteri
    dataframe.drop(drop_indexes, inplace=True)  # Rimozione righe con descrizioni non significative
    print(f"Rimosse {len(drop_indexes)} righe con descrizioni non significative (meno di {des_threshold} caratteri)")



def renameCategories(dataframe, replacementList):
    """
        Metodo che, per mezzo di una lista di roniminazioni, rimpiazza le categorie indicate con i nomi specificati

        Parametri:
        - dataframe, dataframe su cui effettuare l'operazione
        - replacementList, lista di rinominazioni. E' costituita da tuple con il seguente formato:
          (A, B), dove A e B sono due stringhe, per indicare che la categoria A deve essere sostituita con la categoria B
    """
    for replacement in replacementList:
        dataframe["Category"] = dataframe["Category"].str.replace(pat=rf"\s*({replacement[0]})\s*", repl=replacement[1], regex=True)