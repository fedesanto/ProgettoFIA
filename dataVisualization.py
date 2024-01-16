import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)      # Mostrare tutte le colonne
pd.set_option("display.width", None)            # Non tornare a capo quando stampa

def showData(dataframe):
    to_show = 10
    print(f"Prime {to_show} righe:\n")
    print(dataframe.head(to_show))

    print("\nInformazioni generali sul dataframe:\n")
    print(dataframe.info())

    len_threshold = 40
    null_rows = len(dataframe[ dataframe.isna().any(axis=1) ])
    identical_rows = dataframe.duplicated().sum()
    dummy_descriptions = len(dataframe[dataframe["Description"].str.len() < len_threshold])
    print(f"\nNumero di righe con almeno un valore nullo: {null_rows}")
    print(f"Numero righe identiche: {identical_rows}")
    print(f"Righe con descrizioni non significative (meno di {len_threshold} caratteri): {dummy_descriptions}")

    top_categories = 20
    categories = dataframe["Category"].str.split(r"\s*,\s*", expand=True).stack().str.strip()
    category_counts = categories.value_counts()
    top_categories_counts = category_counts.head(top_categories)

    print(f"\nNumero totale di differenti categorie: {len(category_counts)}")
    print(f"\nLe {top_categories} categorie più presenti:")
    print(top_categories_counts)