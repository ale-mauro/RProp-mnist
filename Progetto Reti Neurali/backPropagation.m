function gradiente = backPropagation(net,x,t,funzErr)
%function gradiente = backPropagation(net,x,t,funzErr)
%restituisce il gradiente della funzione di errore (funzErr) rispetto ai pesi e
%rispetto ai bias. 'gradiente' è una struct contenente i campi gradiente.W
%e gradiente.B

%% CONTROLLO INPUT (controllo t sia dimensione (net.c)X(x,2)
if(size(t,1)~=net.c)
    error("Il numero di righe di t ("+size(t,1)+") deve essere uguale a net.c ("+net.c+")");
end
if(size(t,2)~=size(x,2))
    error("Il numero di colonne di t ("+size(t,2)+") deve essere uguale al numero di colonne dell'input x ("+size(x,2)+")");
end

H=net.numLayers; %numero totale di strati
m=length(net.m); %numero strati interni
gradiente.W=cell(1,net.numLayers);
gradiente.B=cell(1,net.numLayers);

%% FASE FORWARD-PROPAGATION(calcolo tutti gli input ed output di tutti i nodi)
[A,Z,y] = forwardStep(net,x);

%% FASE BACK-PROPAGATION 1 (CALCOLO DELTA DI OUTPUT [do=derivFunzActOut*derivFunzErr])
delta_out = derivFunzAct(net.g,A{H});
delta_out= delta_out .* derivFunzErr(funzErr,y,t);

%% FASE BACK-PROPAGATION 2 (CALCOLO DELTA HIDDEN [deltai=derivFunzActHidden*(Wi+1'*deltai+1)])
delta_stratoSucc=delta_out;
for i=m:-1:1 %si parte dall'ultimo strato interno e si arriva al primo (BACK PROPAGANO i delta all'indietro).
    %         delta_hidden{i} = (net.W{i+1})' * delta_stratoSucc;
    %         delta_hidden{i} = delta_hidden{i} .* derivFunzAct(net.f{i}, A{i});
    delta_hidden{i} = derivFunzAct(net.f{i}, A{i}) .* ((net.W{i+1})' * delta_stratoSucc);
    delta_stratoSucc=delta_hidden{i};
end

%% CALCOLO DERIVATE [derv(Wi)=delta(i)*z(i-1)')
z_prev=x;
for i=1:m %calcolo derivate per gli m strati interni
    gradiente.W{i}=delta_hidden{i}*z_prev';
    gradiente.B{i} = sum(delta_hidden{i},2);
    z_prev=Z{i};
end
%calcolo derivata strato output (strato H)
gradiente.W{H} = delta_out*z_prev'; %z_prev corrisponde a l'output del penultimo strato.
gradiente.B{H}= sum(delta_out,2);
end