# RProp-mnist
Studio dell'apprendimento di una rete neurale confrontando le varianti della RProp basato sull'articolo "Empirical evaluation of the improved Rprop learning algorithms - Christian Igel, Michael Husken"

Per eseguire il progetto bisogna eseguire lo script "mainScript".
L'esecuzione dello script farà partire quattro processi di classificazione sul dataset mnist ridotto a 10k immagini, uno per ogni variante.
Al termine dello script sarà mostrato un grafico contenente l'andamento delle quattro varianti corrispondente all'errore sul validation set.
Le variabili contenenti le valutazioni sono: "vettAccTest", "convergenza", "errore" e "tempo". 
Queste variabili sono dei vettori in cui in ogni posizione c'è la valutazione di una variante.
Le prime porzioni di codice sono dedicate alla scelta degli iperparametri, è possibile cambiare iperparametri a proprio piacimento.

Lo script "sceltaIperparametri" permette di trovare la configurazione migliore degli iperparametri della variante RProp+.
Siccome la computazione impiega del tempo, nella cartella "variables" sono contenute le variabili "matriceRisultatiMedia" e "matriceRisultatiTotale", 
ovvero le variabili che si ottengono alla fine della computazione.

# LE FUNZIONI IMPLEMENTATE SONO LE SEGUENTI:
newNet.m
simNet.m / forwardStep.m
backPropagation.m
RPROP.m
learningPhase.m
mainScript.m

sceltaIperparametri.m
derivFunzErr.m
derivFunzAct.m
accuracy.m
discesaDelGradienteStandard.m
softMax.m

# FUNZIONI DI ERRORE E DERIVATE:
crossEntropyMC.m
derivCrossEntropyMC.m

crossEntropySoftMax.m
derivCrossEntropySoftMax.m

sumOfSquares.m
derivFunzErrSumOfSquares.m


# FUNZIONI DI ATTIVAZIONE E DERIVATE:
identity.m
derivFunzActIdentity.m

sigmoide.m
derivFunzActSigmoide.m


# FUNZIONI PER DATASET MNIST:
getTargetsFromLabels.m
loadMNISTImages.m
loadMNISTLabels.m
