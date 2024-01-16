import pandas as pd
import numpy as np
from dataVisualization import *
from dataPreparation import *

df = pd.read_csv("Data/BooksDataset.csv", usecols=["Title", "Description", "Authors", "Category"])

showData(df)

print("\n------------")
print("Pulizia dataset")
cleanData(df)
print("------------")

replacementList = [
    ("Religious", "Religion"),
    ("Christian Life", "Religion"),
    ("Historical", "History"),
    ("Family", "Family & Relationships"),
    ("Pets", "Animal")
]
renameCategories(df, replacementList)

