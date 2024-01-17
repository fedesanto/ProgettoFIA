import pandas as pd
import numpy as np
import nltk
nltk.data.path.append("nltk_data")

from dataVisualization import *
from dataPreparation import *


df = pd.read_csv("Data/BooksDataset.csv", usecols=["Title", "Description", "Authors", "Category"])

showData(df)

print("\n------------")
print("Pulizia dataset")
cleanData(df)
print("------------")

print("\n------------")
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
print("------------")

print("\n------------")
extractList = ["Religion", "Romance", "Cooking", "History",
               "Business & Economics", "Thrillers", "Mystery & Detective",
               "Health & Fitness", "Art",  "Sports & Recreation", "Travel",
               "Fantasy", "Science", "Animals", "Computers", "House & Home"]
print("Estrazione dei libri appartenenti alle seguenti categoire:")
for category in extractList:
    print(category)

extractCategories(df, extractList)
print("------------")

print("\n------------")
print("Bilanciamento del dataset")
balanceCategories(df, 2800)
print("------------")

print("\n------------")
description_stopwords = stopwords.words('english') + ["life", "book", "one", "new", "time", "world", "find"]
authors_stopwords = ["book", "books", "editor", "editors", "mr", "mrs", "dr", "jr", "magazine", "and"]

print("Preprocessing delle descrizioni...")
preprocessDescription(df, description_stopwords)

print("Preprocessing degli autori...")
preprocessAuthors(df, authors_stopwords)
print("------------")

showData(df)

