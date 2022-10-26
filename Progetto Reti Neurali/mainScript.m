%Al termine dello script, le variabili contenenti le valutazioni sono:
%vettAccTest(methNum)=accuracy(simNet(netScelta,XTest),TTest);
%convergenza(methNum)=ind_minErr;
%errore(methNum)=funzErr(simNet(netScelta,XTest),TTest)/10000;

%% IPERPARAMETRI
f={@sigmoide};
g=@identity;

eta=0.0005; %ottimale

%iperparametri standard
% M=150;
% eta_p=1.2;
% eta_n=0.5;

%iperparametri migliori
M=250;
eta_p=1.05;
eta_n=0.4;

funzErr=@crossEntropySoftMax;

MAX_EPOCHES=55;
%Si scelgono le varianti da voler eseguire, con i rispettivi colori sul
%grafico.
methods={'irprop-',  'rprop+', 'irprop+','irprop-'};
colors= {'red',     'blue',   'green',   'black'};
%% CARICAMENTO DATASET
if not(exist('X'))
    X=loadMNISTImages('mnist/t10k-images-idx3-ubyte');
    Labels=loadMNISTLabels('mnist/t10k-labels-idx1-ubyte');
    %     X=loadMNISTImages('mnist/train-images-idx3-ubyte');
    %     Labels=loadMNISTLabels('mnist/train-labels-idx1-ubyte');
    T=getTargetsFromLabels(Labels);
end


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

for methNum=1:length(methods)
    tic;
    method=methods{methNum};
    color=colors{methNum};
    %         method='irprop-';
    disp(method);
    
    %% LEARNING PHASE
    net=newNet(size(XTrain,1),M,size(TTrain,1),f,g);
    [netScelta,errTrain,errVal,accTrain,accVal]=...
        learningPhase(net,XTrain,TTrain,XVal,TVal,funzErr,MAX_EPOCHES,eta,eta_p,eta_n,method);
    
    %% Valutazione rete scelta dalla fase di learning
    [min_err,ind_minErr]=min(errVal);
    vettAccTest(methNum)=accuracy(simNet(netScelta,XTest),TTest);
    convergenza(methNum)=ind_minErr;
    errore(methNum)=funzErr(simNet(netScelta,XTest),TTest)/10000;
    tempo(methNum)=toc;
    %% Plot
    paramaters=strcat(...
        'M=',num2str(M),...
        '  \eta^+=', num2str(eta_p),...
        '  \eta^-=', num2str(eta_n),...
        '  \Delta_{max}=50',...
        '  \Delta_{min}=0');
    %figure;
    %plot(1:MAX_EPOCHES,errTrain,'r-o');
    hold on;
    me=plot(1:MAX_EPOCHES,errVal,color,'LineWidth',1.5);
    xlabel('Epoche');
    ylabel('Error');
    title(strcat('Confronto valutazioni varianti RProp su validation set'), paramaters);
end
legend(methods);
hold off