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

from time import time

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


def getModelParams(model = "LinearSVC"):
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

        case _:
            params = {"transformer__aut_transformer__ngram_range": [(1, 1), (1, 2), (2, 2)],
                      "clf__penalty": ["l1", "l2"],
                      "clf__C": [1, 10, 100, 1000],
                      "clf__tol": [0.001, 0.0001, 0.00001]
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