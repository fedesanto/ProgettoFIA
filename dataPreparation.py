import pandas as pd
import numpy as np
import nltk

from string import punctuation
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords

nltk.data.path.append("nltk_data")  # Necessario per indicare alla libreria "nltk" dove si trovano i dati di cui ha bisogno


def cleanData(dataframe, des_threshold = 40):
    """
        Funzione per la pulizia del dataframe. In successione:
            1) Rimuove righe con almeno un campo nullo
            2) Rimuove righe duplicate
            3) Rimuove righe con descrizioni non significative (con meno di 'des_threshold' caratteri)

        Parametri:
            - dataframe, dataframe su cui effetture la pulizia
            - des_threshold, numero di caratteri minimo che una descrizione deve avere per considerarsi significativa
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
        Funzione che, per mezzo di una lista di rinominazioni, rimpiazza le categorie indicate con i nomi specificati

        Parametri:
            - dataframe, dataframe su cui effettuare l'operazione
            - replacementList, lista di rinominazioni. E' costituita da tuple con il seguente formato:
              (A, B), dove A e B sono due stringhe, per indicare che la categoria A deve essere sostituita con la categoria B
    """
    for replacement in replacementList:
        dataframe["Category"] = dataframe["Category"].str.replace(pat=rf"\s*({replacement[0]})\s*", repl=replacement[1], regex=True)


def extractCategories(dataframe, toExtract):
    """
        Funzione di estrazione delle categorie di interesse e rimozione delle righe non appartenenti a nessuna delle categorie indicate.
        Per ogni riga:
            1) Estrae tutte le categorie separate da ','
            2) Se nessuna delle categorie estratte appartiene a 'toExtract', allora la riga viene eliminata
            3) altrimenti, tra quelle categorie che fanno parte di 'toExtact', viene assegnato al campo "Category" quella che ha meno esempi nel dataframe

        Parametri:
            - dataframe, dataframe su cui effetture la pulizia
            - toExtract, lista di stringhe rappresentanti le categorie da estrarre
    """
    categories = dataframe["Category"].str.split(r"\s*,\s*", expand=True).stack()  # Ottengo la lista di tutte le categorie
    categories = categories.str.strip()  # Rimuovo spazi vuoti superflui
    categories = categories.str.lower()  # Mi assicuro che tutte le lettere siano in minuscolo
    toExtract = [cat.strip().lower() for cat in toExtract]

    categories_counts = categories.value_counts()    # Ottengo il conteggio di ciascuna categoria
    drop_indexes = []  # Lista degli indici delle righe che saranno droppate dal dataframe

    for index, row in dataframe.iterrows():
        categories_splitted = [splitted.strip().lower() for splitted in row["Category"].split(",")]   # Leggo il campo "Category" ed estraggo tutte le categorie

        changed = False
        for cat in categories_splitted:   # Per ogni categoria della riga
            if (cat in toExtract) and (cat in categories_counts):          # Verifico se la categoria è presente nell'elenco delle categorie da estrarre
                if (not changed) or (categories_counts[cat] < categories_counts[row["Category"]]):
                    row["Category"] = cat
                    changed = True

        if changed:     # Se la riga ha subito un cambiamento allora sovrascivo il cambiamento avvenuto al dataframe
            dataframe["Category"].loc[index] = row["Category"]
        else:           # altrimenti considero la riga da eliminare, in quanto non appartiene a nessuna delle categorie di interesse
            drop_indexes.append(index)

    dataframe.drop(drop_indexes, inplace=True)      # Elimino le righe considerate da scartare



def balanceCategories(dataframe, threshold):
    """
        Funzione di bilanciamento delle categorie.
        Verifica il numero di righe per ogni categoria e elimina casualmente delle righe di categorie con più di 'threshold' esempi

        Parametri:
            - dataframe, dataframe su cui effettuare l'operazione
            - threshold, numero limite di righe che una categoria deve avere
    """
    drop_indexes = []

    for category in dataframe["Category"].unique():
        remove_n = len(dataframe[dataframe["Category"] == category]) - threshold

        if remove_n > 0:
            indexes = np.random.choice(dataframe[dataframe["Category"] == category].index, remove_n, replace=False)
            drop_indexes.extend(indexes)
            print(f"Rimosse {remove_n} righe dalla categoria {category}")

    dataframe.drop(drop_indexes, inplace=True)


def preprocessDescription(dataframe, stop_words = stopwords.words('english')):
    """
        Funzione di preprocessing delle descrizioni.
        In ordine:
            1) Rende minuscole tutte le lettere
            2) Rimuove la punteggiature
            3) Rimuove le stopwords passate in 'stop_words'
            4) Effettua la lemmatizzazione delle parole

        Parametri:
            - dataframe, dataframe su cui effettuare l'operazione
            - stop_words, lista di stringhe rappresentanti le stopwords da eliminare
    """
    newDescriptions = []
    for index, row in dataframe.iterrows():
        text = row["Description"]

        text = text.lower()  # Ogni lettere viene portata in minuscolo

        text = "".join([char for char in text if char not in punctuation])  # Rimozione della punteggiature

        words = word_tokenize(text)
        filtered_words = [word for word in words if word not in stop_words]  # Rimozione stopwords

        lemmatizer = WordNetLemmatizer()
        lemmed = [lemmatizer.lemmatize(word) for word in filtered_words]  # Lemming

        newDescriptions.append(" ".join(lemmed))

    dataframe["Description"] = newDescriptions


def preprocessAuthors(dataframe, stop_words = []):
    """
        Funzione di preprocessing del campo autori.
        In ordine:
            1) Rende minuscole tutte le lettere
            2) Rimuove la punteggiature
            3) Rimuove le stopwords passate in 'stop_words' ed eventuali caratteri singoli

        Parametri:
            - dataframe, dataframe su cui effettuare l'operazione
            - stop_words, lista di stringhe rappresentanti le stopwords da eliminare
    """
    newAuthors = []
    if "by" not in stop_words:
        stop_words.append("by")

    editors = dataframe["Authors"].str.extract(r"\((\w+)\)", expand=False).dropna().unique().tolist()  # Lista delle case editrici tra parentesi
    editors = [word.lower() for word in editors]
    for word in editors:
        if word not in stop_words:
            stop_words.append(word)

    for index, row in dataframe.iterrows():
        text = row["Authors"]

        text = text.lower()  # Ogni lettere viene portata in minuscolo

        text = "".join([char for char in text if char not in punctuation])  # Rimozione punteggiatura

        words = word_tokenize(text)
        filtered_words = [word for word in words if word not in stop_words and len(word) > 1]  # Rimozione stopwords e caratteri singoli

        newAuthors.append(" ".join(filtered_words))

    dataframe["Authors"] = newAuthors