Il nostro progetto si pone come obiettivo lo studio e la sperimentazione di differenti tecniche di Machine Learning capaci di analizzare ed estrarre informazioni da dati sotto forma di linguaggio naturale. Nello specifico si è interessati alla categorizzazione di libri tramite una breve descrizione testuale ed un elenco di autori. La categorizzazione è stata elaborata tramite due tecniche di Machine Learning, Classificazione e Clustering, che, seppur facenti utilizzo di due approcci diversi (rispettivamente apprendimento supervisionato e apprendimento non supervisionato), in linea teorica dovrebbero essere in grado di riportare risultati similari e confrontabili. A tal proposito occorrerà addestrare più modelli differenti per entrambe le tecniche e verificare quali di questi permette di ottenere il miglior risultato.

In questa repository è possibile accedere alle seguenti risorse:
 - Documentazione ufficiale del progetto
 - Dataset su cui è stato basato l'addestramento
 - Plots per l'analisi del dataset e per la valutazione dei risultati ottenuti dai vari modelli
 - Codice per la visualizzazione ed elaborazione dei dati e l'addrestramento e valutazione dei modelli di machine learning utilizzati
 - Script della demo per provare uno dei classificatori già addestrati messi a disposizione

Per replicare il lavoro da noi svolto è sufficiente avviare lo script "main.py", il quale si occupa autonomamente di mostrare alcune informazioni utili sul dataset, prepararlo alla fase di addestramento ed, infine, addestrare differenti modelli di classificazione e di clustering, mostrando poi i risultati ottenuti sotto forma di output testuale e plots grafici.
Tramite l'avvio dello script "mainDemo.py" è possibile effettuare una demo dei modelli di classificazione già addestrati e messi a disposzione, richiedendo all'utente: (1) quale dei modelli si desidera utilizzare, (2) titolo, decrizione e autori dei libri che si intende predirre. Infine vengono mostrare le predizioni delle categorie per ciascun libro.
