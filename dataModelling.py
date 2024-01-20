import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering
from sklearn.metrics import silhouette_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV

from sklearn.decomposition import PCA
import seaborn as sns

from time import time

def trainClusters(X, model, findBestEstimator = False, returnFitTime = True):
    """
        Funzione che permette l'addestramento di un modello di cluster a sceltra tra KMeans, MiniBatchKMeans e SpectralClustering

        Parametri:
               - X, dataframe con due colonne di "Description" e "Authors"
               - model, stringa che indica il tipo di modello da addestrare (KMeans, MiniBatchKMeans e SpectralClustering)
               - findBestEstimator, se True vengono addestrati in maniera casuale più classficatori con parametri differenti
                 e ne si restituisce il migliore
               - returnFitTime, se True viene restuito il tempo impiegato per addestrare il modello
    """

    description_vect = TfidfVectorizer()
    authors_vect = TfidfVectorizer(ngram_range=(1, 2))
    colTransformer = ColumnTransformer([
        ('des_transformer', description_vect, "Description"),
        ('aut_transformer', authors_vect, "Authors")
    ])

    K_clusters = 16     #si scelgono 16 clusters in seguito alle analisi fatte sulle categorie del dataframe

    if model == "KMeans":
        modelPipe = Pipeline([('transformer', colTransformer), ('clt', KMeans(n_clusters = K_clusters, random_state = 42, n_init = 5))])
    elif model == "MiniBatchKMeans":
        modelPipe = Pipeline([('transformer', colTransformer), ('clt', MiniBatchKMeans(n_clusters=K_clusters, random_state=42, n_init=5))])
    elif model == "SpectralClustering":
        modelPipe = Pipeline([('transformer', colTransformer), ('clt', SpectralClustering(n_clusters=K_clusters, random_state=42, n_init=5))])
    else:
        raise Exception("Inserisci il nome di un modello valido, a scelta tra: KMeans, MiniBatchKMeans e SpectralClustering")

    if findBestEstimator:
        params = getModelParams(model)
        rs = RandomizedSearchCV(modelPipe, params, cv=5)

        start_time = time()
        rs.fit(X)
        fitTime = time() - start_time

        if returnFitTime:
            return rs.best_estimator_, fitTime
        else:
            return rs.best_estimator_
    else:
        startTime = time()
        modelPipe.fit(X)
        fitTime = time() - startTime

        if returnFitTime:
            return modelPipe, fitTime
        else:
            return modelPipe


def trainClassificator(X_train, Y_train, model = "LinearSVC", findBestEstimator = False, returnFitTime = True):
    description_vect = TfidfVectorizer()
    authors_vect = TfidfVectorizer(ngram_range=(2, 2))
    colTransformer = ColumnTransformer([
        ('des_transformer', description_vect, "Description"),
        ('aut_transformer', authors_vect, "Authors")
    ])

    match model:
        case "LogisticRegression":
            modelPipe = Pipeline([('transformer', colTransformer), ('clf', LogisticRegression())])

        case "SGDClassifier":
            modelPipe = Pipeline([('transformer', colTransformer), ('clf', SGDClassifier())])

        case "MultinomialNB":
            modelPipe = Pipeline([('transformer', colTransformer), ('clf', MultinomialNB())])

        case _:
            modelPipe = Pipeline([('transformer', colTransformer), ('clf', LinearSVC())])

    if findBestEstimator:
        params = getModelParams(model)
        rs = RandomizedSearchCV(modelPipe, params, cv=5)

        start_time = time()
        rs.fit(X_train, Y_train)
        fitTime = time() - start_time

        if returnFitTime:
            return rs.best_estimator_, fitTime
        else:
            return rs.best_estimator_
    else:
        startTime = time()
        modelPipe.fit(X_train, Y_train)
        fitTime = time() - startTime

        if returnFitTime:
            return modelPipe, fitTime
        else:
            return modelPipe


def getModelParams(model):
    match model:
        case "LogisticRegression":
            params ={"transformer__aut_transformer__ngram_range": [(1, 1), (1, 2), (2, 2)],
                     "clf__solver": ["sag", "saga", "lbfgs", "newton-cg"],
                     "clf__C": [1, 10, 100, 1000],
                     "clf__tol": [0.001, 0.0001, 0.00001]
                     }

        case "SGDClassifier":
            params ={"transformer__aut_transformer__ngram_range": [(1, 1), (1, 2), (2, 2)],
                     "clf__loss": ["hinge", "log_loss", "squared_hinge"],
                     "clf__penalty": ["l2", "l1", "elasticnet"],
                     "clf__alpha": [0.001, 0.0001, 0.00001],
                     "clf__tol": [0.001, 0.0001, 0.00001]
                     }

        case "MultinomialNB":
            params ={"transformer__aut_transformer__ngram_range": [(1, 1), (1, 2), (2, 2)]
                     }

        case "LinearSVC":
            params = {"transformer__aut_transformer__ngram_range": [(1, 1), (1, 2), (2, 2)],
                      "clf__penalty": ["l1", "l2"],
                      "clf__C": [1, 10, 100, 1000],
                      "clf__tol": [0.001, 0.0001, 0.00001]
                      }

        case "KMeans":
            params = {"transformer__aut_transformer__ngram_range": [(1, 1), (1, 2), (2, 2)],
                      "clf__init": ["k-means++", "random"],
                      "clf__tol": [0.001, 0.0001, 0.00001],
                      "clf__algorithm": ["lloyd", "elkan"]
                      }

        case "MiniBatchKMeans":
            params = {"transformer__aut_transformer__ngram_range": [(1, 1), (1, 2), (2, 2)],
                      "clf__init": ["k-means++", "random"],
                      "clf__batch_size": [256, 512, 1024, 2048],
                      "clf__tol": [0.001, 0.0001, 0.00001],
                      "clf__max_no_improvement": [10, 20, 30],
                      "clf__init_size": [3072, 3500, 4000],
                      "clf__reassignment_ratio": [0.01, 0.05 ,0.1]
                      }

        case "SpectralClustering":
            params = {"transformer__aut_transformer__ngram_range": [(1, 1), (1, 2), (2, 2)],
                      "clf__eigen_solver": ["arpack", "lobpcg"],
                      "clf__affinity": ["nearest_neighbors", "rbf", "precomputed", "precomputed_nearest_neighbors"],
                      "clf__assign_labels": ["kmeans", "discretize", "cluster_qr"]
                      }


    return params


def testClassificator(X_test, Y_test, model, returnPredictionTime=True, saveConfusionMatrix=None):
    startTime = time()
    prediction = model.predict(X_test)
    predictionTime = time() - startTime

    print("Accuracy score: ", round(accuracy_score(Y_test, prediction), 3))
    print("\nClassification report:\n")
    print(classification_report(Y_test, prediction))

    if saveConfusionMatrix:
        cm = confusion_matrix(Y_test, prediction, labels=model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot()
        plt.xticks(rotation=90)
        plt.savefig(f"Plots/{saveConfusionMatrix}.png")
        print(f"Memorizzata la matrice di confusione in: /Plots/{saveConfusionMatrix}.png")

    if returnPredictionTime:
        return predictionTime


def clusterAnalysis(X,model,n_terms=10):
    """
        Funzione che permette di calcolare il Silhouette Score di un insieme di cluster e di
        visualizzare le keywords delle descrizioni e degli autori per ogni cluster

        Parametri:
            - X, dataframe con due colonne di "Description" e "Authors"
            - model, pipeline contenente il columnTransformer e la metodologia di clustering da analizzare
            - n_terms, numero delle stringhe principali da visualizzare per ogni cluster
       """

    transformer = model.named_steps["transformer"]
    des_transformer = transformer.named_transformers_["des_transformer"]
    aut_transformer = transformer.named_transformers_["aut_transformer"]

    X_transformed = transformer.transform(X)
    X_description = des_transformer.transform(X["Description"])
    X_authors = aut_transformer.transform(X["Authors"])

    clusters = model.named_steps["clt"].labels_
    print(f"Silhouette Score: {silhouette_score(X_transformed, clusters)}")

    print("Le keywors delle descrizioni per ogni cluster sono:")
    # Identificazione delle keywords nelle descrizioni per ogni cluster
    data = pd.DataFrame(X_description.todense()).groupby(clusters).mean()  # raggruppa il vettore TF-IDF per gruppo
    terms = des_transformer.get_feature_names_out()  # accedi ai termini del tf idf
    for i, r in data.iterrows():
        print(f'\nCluster {i}')
        print(','.join([terms[t] for t in np.argsort(r)][::-1][:n_terms]))  # per ogni riga del dataframe, trova gli n termini che hanno il punteggio più alto

    print("\nGli autori più comuni per ogni cluster sono:")
    # Identificazione degli autori più comuni per ogni cluster
    data = pd.DataFrame(X_authors.todense()).groupby(clusters).mean()  # raggruppa il vettore TF-IDF per gruppo
    terms = aut_transformer.get_feature_names_out()  # accedi ai termini del tf idf
    for i, r in data.iterrows():
        print(f'\nCluster {i}')
        print(','.join([terms[t] for t in np.argsort(r)][::-1][:n_terms]))  # per ogni riga del dataframe, trova gli n termini che hanno il punteggio più alto

    # inizializziamo la PCA con 2 componenti
    pca = PCA(n_components=2, random_state=42)
    # passiamo alla pca il nostro array X
    pca_vecs = pca.fit_transform(X_transformed.toarray())
