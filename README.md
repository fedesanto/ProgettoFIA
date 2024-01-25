Il nostro progetto si pone come obiettivo lo studio e la sperimentazione di differenti tecniche di Machine Learning capaci di analizzare ed estrarre informazioni da dati sotto forma di linguaggio naturale. Nello specifico si è interessati alla categorizzazione di libri tramite una breve descrizione testuale ed un elenco di autori. La categorizzazione è stata elaborata tramite due tecniche di Machine Learning, Classificazione e Clustering, che, seppur facenti utilizzo di due approcci diversi (rispettivamente apprendimento supervisionato e apprendimento non supervisionato), in linea teorica dovrebbero essere in grado di riportare risultati similari e confrontabili. A tal proposito occorrerà addestrare più modelli differenti per entrambe le tecniche e verificare quali di questi permette di ottenere il miglior risultato.

In questa repository è possibile accedere alle seguenti risorse:
 - Documentazione ufficiale del progetto
 - Dataset su cui è stato basato l'addestramento
 - Plots per l'analisi del dataset e per la valutazione dei risultati ottenuti dai vari modelli
 - Codice per la visualizzazione ed elaborazione dei dati e l'addestramento e valutazione dei modelli di machine learning utilizzati
 - Script della demo per provare uno dei classificatori già addestrati messi a disposizione

Per replicare il lavoro da noi svolto è sufficiente avviare lo script "main.py", il quale si occupa autonomamente di mostrare alcune informazioni utili sul dataset, prepararlo alla fase di addestramento e, infine, addestrare differenti modelli di classificazione e di clustering, mostrando poi i risultati ottenuti sotto forma di output testuale e plots grafici. Si tenga presente che gli addestramenti dei modelli possono impiegare significativo tempo, soprattutto durante la fase di ottimizzazione (dai 10 ai 15 minuti). Inoltre, per alcune fasi è richiesto un gran quantitativo di memoria e, nel caso in cui il dispositivo su cui si sta provando lo script non la disponesse, l'esecuzione potrebbe fallire.

Tramite l'avvio dello script "mainDemo.py" è invece possibile effettuare una demo dei modelli di classificazione già addestrati e messi a disposizione, richiedendo all'utente: (1) quale dei modelli si desidera utilizzare, (2) titolo, descrizione e autori dei libri che si intende predire. Infine, vengono mostrare le predizioni delle categorie per ciascun libro.

Per l'esecuzione degli script potrebbe essere necessario installare sul proprio interprete Python alcune librerie utilizzate, tra cui:
 - pandas, versione 2.2.0
 - scikit-learn, versione 1.4.0rc1
 - numpy, versione 1.26.3
 - matplotlib, versione 3.8.2
 - joblib, versione 1.3.2
 - nltk, versione 3.8.1
 - wordcloud, versione 1.9.3
