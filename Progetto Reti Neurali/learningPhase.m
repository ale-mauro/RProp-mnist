function [netScelta,errTrain,errVal,accTrain,accVal]=learningPhase(net,x,t,x_val,t_val,funzErr,maxEpochs,eta,eta_p,eta_n,method)
%function [netScelta,errTrain,errVal,accTrain,accVal]=learningPhase(net,x,t,x_val,t_val,funzErr,maxEpochs,eta,eta_p,eta_n,method)
%Funzione che ritorna la rete che generalizza meglio (che ha minor errore
%su validation set). Ritorna anche l'errore e l'accuratezza sia sul training set che sul validation set.

%% INIZIALIZZAZIONE VALORI
errTrain=zeros(1,maxEpochs);
errVal=zeros(1,maxEpochs);
accTrain=zeros(1,maxEpochs);
accVal=zeros(1,maxEpochs);
oldGrad=0;
for i=1:net.numLayers
    %     rangeNumber=0.15;
    %     Delta.W{i}=rand(size(net.W{i})).*rangeNumber;
    %     Delta.B{i}=rand(size(net.B{i})).*rangeNumber;
    Delta.W{i}=zeros(size(net.W{i}))+0.0125;
    Delta.B{i}=zeros(size(net.B{i}))+0.0125;
    Delta.max.W{i}=zeros(size(net.W{i}))+50;
    Delta.min.W{i}=zeros(size(net.W{i}));
    Delta.max.B{i}=zeros(size(net.B{i}))+50;
    Delta.min.B{i}=zeros(size(net.B{i}));
end

for i=1:net.numLayers
    oldMod.W{i}=zeros(size(Delta.W{i}));
    oldMod.B{i}=zeros(size(Delta.B{i}));
end

y_val=simNet(net,x_val);
min_err=funzErr(y_val,t_val);
netScelta=net;
method=lower(method);
%% EPOCHE DI APPRENDIMENTO
for epoch=1:maxEpochs %In QUESTO CASO sto suppenendo di fare sempre tutte le iterazioni
    %% AGGIORNAMENTO PARAMETRI
    gradiente=backPropagation(net,x,t,funzErr);
    [net,Delta,oldMod,gradiente]=RPROP(net,method,eta,eta_p,eta_n,gradiente,oldGrad,Delta,oldMod,errTrain,epoch);
    %net=discesaDelGradienteStandard(net,eta, gradiente);
    oldGrad=gradiente;
    %% VALUTAZIONE RETE
    y=simNet(net,x);
    y_val=simNet(net,x_val);
    
    errTrain(epoch)=funzErr(y,t)/10000;
    errVal(epoch)=funzErr(y_val,t_val)/10000;
    
    accTrain(epoch)=accuracy(y,t);
    accVal(epoch)=accuracy(y_val,t_val);
    
    disp(['epoch' num2str(epoch) '_ err train: ' num2str(errTrain(epoch)) '; err val: ' num2str(errVal(epoch)) '; accTrain: ' num2str(accTrain(epoch)) '; accVal: ' num2str(accVal(epoch))]);
    
    %% CALCOLO RETE CON ERRORE MINIMO
    if errVal(epoch)< min_err
        min_err=errVal(epoch);
        netScelta=net;
    end
end
%netScelta=net;
%In questo caso restituisco la rete ottenuta all'ultima epoca
end