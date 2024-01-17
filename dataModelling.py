from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import SpectralClustering
from time import time
import pandas as pd
import numpy as np

def trainClusters(dataframe,model,n_terms=10):
    """
        Funzione che permette l'addestramento di un modello di cluster a sceltra tra KMeans, MiniBatchKMeans e SpectralClustering,
        con relativo Silhouette Score e visualizzazione delle keywords per ogni cluster

        Parametri:
               - dataframe, dataframe su cui effetture la pulizia
               - model, stringa che indica il tipo di modello da addestrare (Means, MiniBatchKMeans e SpectralClustering)
               - n_terms, numero di keywords da visualizzare per ogni cluster
    """

    vectorizer = TfidfVectorizer()
    X= vectorizer.fit_transform(dataframe['Description'])
    K_clusters = 16     #si scelgono 16 clusters in seguito alle analisi fatte sulle categorie del dataframe

    if model == "KMeans":
        kMeans = KMeans(n_clusters = K_clusters, random_state = 42, n_init = 5)
        print("Addestramento del modello...")
        act = time()
        KMeans.fit(X)
        print(f"Addestramento completato in {time()-act}s")
        clusters = kMeans.labels_
    elif model == "MiniBatchKMeans":
        MBKMeans = MiniBatchKMeans(n_clusters=K_clusters, random_state=42, n_init=5)
        print("Addestramento del modello...")
        act = time()
        MBKMeans.fit(X)
        print(f"Addestramento completato in {time() - act}s")
        clusters = MBKMeans.labels_
    elif model == "SpectralClustering":
        SCluster = SpectralClustering(n_clusters=K_clusters, random_state=42, n_init=5)
        print("Addestramento del modello...")
        act = time()
        SCluster.fit(X)
        print(f"Addestramento completato in {time() - act}s")
        clusters = SCluster.labels_
    else:
        print("Inserisci il nome di un modello valido, a scelta tra: KMeans, MiniBatchKMeans e SpectralClustering")

    print(f"Silhouette Score: {silhouette_score(X, clusters)}")

    #Identificazione delle keywords per ogni cluster
    data = pd.DataFrame(X.todense()).groupby(clusters).mean()  # raggruppa il vettore TF-IDF per gruppo
    terms = vectorizer.get_feature_names_out()  # accedi ai termini del tf idf
    for i, r in data.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([terms[t] for t in np.argsort(r)][::-1][:n_terms]))  # per ogni riga del dataframe, trova gli n termini che hanno il punteggio più alto