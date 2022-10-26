%Costruisce una matrice 'matriceRisultatiTotale', ovvero un cell array in
%cui ogni cella contiene le valutazioni di ogni configurazione degli
%iperparametri. Inoltre costruisce la matrice 'matriceRisultatiMedia'
%contenente i risultati medi di tutte le configurazioni. Gli iperparametri
%migliori sono scelti ordinando 'matriceRisultatiMedia' rispetto all'errore

%% IPERPARAMETRI
f={@sigmoide};
g=@identity;
eta=0.0005; %ottimale

funzErr=@crossEntropySoftMax;
MAX_EPOCHES=100;

method='rprop+';
VM=[150,170,200,220,250];
Veta_p=[1.2, 1.15, 1.1, 1.07, 1.05];
Veta_n=[0.5, 0.4, 0.3, 0.25, 0.2];
%% CARICAMENTO DATASET
if not(exist('X'))
    X=loadMNISTImages('mnist/t10k-images-idx3-ubyte');
    Labels=loadMNISTLabels('mnist/t10k-labels-idx1-ubyte');
    T=getTargetsFromLabels(Labels);
end

for k=1:10
    %% SHUFFLE
    ind=randperm(size(X,2));
    X=X(:,ind);
    T=T(:,ind);
    
    %% SUDDIVISIONE DATASET IN TRAINING, VALIDATION E TEST
    half=size(X,2)/2;
    three_quarter=half+size(X,2)/4;
    
    XTrain= X(:,1:half);
    TTrain= T(:,1:half);
    
    XVal=X(:,half+1:three_quarter);
    TVal= T(:,half+1:three_quarter);
    
    XTest=X(:,three_quarter+1:end);
    TTest= T(:,three_quarter+1:end);
    
    %%
    for ind_M=1:length(VM)
        M=VM(ind_M);
        for ind_etap=1:length(Veta_p)
            eta_p=Veta_p(ind_etap);
            for ind_etan=1:length(Veta_n)
                eta_n=Veta_n(ind_etan);
                riga=(ind_M-1)*25+(ind_etap-1)*5+ind_etan;
                disp([k, riga]);
                net=newNet(size(XTrain,1),M,size(TTrain,1),f,g);
                [netScelta,errTrain,errVal,accTrain,accVal]=...
                    learningPhase(net,XTrain,TTrain,XVal,TVal,funzErr,MAX_EPOCHES,eta,eta_p,eta_n,method);
                
                %% Valutazione rete scelta dalla fase di learning
                accTest=accuracy(simNet(netScelta,XTest),TTest);
                [min_err,convergenza]=min(errVal);
                erroreTest=funzErr(simNet(netScelta,XTest),TTest)/10000;
                
                matriceRisultatiTotale{k}(riga,:)=[M,eta_p,eta_n, accTest, erroreTest,convergenza];
                
            end
        end
    end
end


for i=1:10
    accuracys(:,i)=matriceRisultatiTotale{i}(:,4);
    errors(:,i)=matriceRisultatiTotale{i}(:,5);
    convergenzes(:,i)=matriceRisultatiTotale{i}(:,6);
end

for i=1:125
    medieAccuracy(i,:)=mean(accuracys(i,:));
    devStdAccuracy(i,:)=std(accuracys(i,:));
    
    medieErrore(i,:)=mean(errors(i,:));
    devStdErrore(i,:)=std(errors(i,:));
    
    medieConvergenze(i,:)=mean(convergenzes(i,:));
    devStdConvergenze(i,:)=std(convergenzes(i,:));
end

matriceRisultatiMedia=matriceRisultatiTotale{1};
matriceRisultatiMedia(:,4)=medieAccuracy;
matriceRisultatiMedia(:,5)=devStdAccuracy;
matriceRisultatiMedia(:,6)=medieErrore;
matriceRisultatiMedia(:,7)=devStdErrore;
matriceRisultatiMedia(:,8)=medieConvergenze;
matriceRisultatiMedia(:,9)=devStdConvergenze;

% [m,i]=max(matriceRisultatiMedia(:,4));
% hyperparameter_1=matriceRisultatiMedia(i,1:3);
%
% [m,i]=min(matriceRisultatiMedia(:,5));
% hyperparameter_2=matriceRisultatiMedia(i,:);
%
[m,i]=min(matriceRisultatiMedia(:,6));
hyperparameter=matriceRisultatiMedia(i,1:3);
%
% [m,i]=min(matriceRisultatiMedia(:,7));
% hyperparameter_4=matriceRisultatiMedia(i,1:3);
%
% [m,i]=min(matriceRisultatiMedia(:,8));
% hyperparameter_5=matriceRisultatiMedia(i,1:3);
%
% [m,i]=min(matriceRisultatiMedia(:,9));
% hyperparameter_6=matriceRisultatiMedia(i,1:3);

