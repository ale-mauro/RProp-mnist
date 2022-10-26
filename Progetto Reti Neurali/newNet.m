function net=newNet(d,m,c,f,g)
%net=newNet(d,m,c,f,g)
%Creazione di una net full connected multistrato con qualsiasi numero di strati interni
%e qualsiasi funzione di attivazione (la lunghezza di m determina il numero di strati interni).

%% CONTROLLO INPUT
assert(nargin==5,"Inserisci il numero corretto di parametri (5 parametri)");
assert(mod(d,1)==0 && all(mod(m,1)==0) && mod(c,1)==0, "d,m,c devono essere numeri decimali");
assert(d>0 && all(m>0) && c>0,"I parametri d,m,c devono essere maggiori di 0");
assert(size(f,2)==1 || size(f,2)==size(m,2),"Bisogna assegnare una funzione di attivazione per ogni strato. Inserisci solo una per assegnare la stessa funzione a tutti gli strati");
assert(isa(f,'cell'),"le funzioni di errore degli strati interni devono essere memorizzati in un cell array");
for i=1:size(f,2)
    assert(isa(f{i},'function_handle'),"Le funzioni di attivazione degli strati di input devono essere function_handle");
end
assert(isa(g,'function_handle'),"La funzione di attivazione dello strato di output deve essere un function_handle");

%% INIZIALIZZAZIONE CELL ARRAYS
len=length(m)+1;
net.W=cell(1,len);
net.B=cell(1,len);
if size(f,2)==1 && length(m)>1
    for i=1:length(m)
        f{i}=f{1};
    end
end

%% CREAZIONE STRATI INTERNI
SIGMA=0.2; prec_layer=d;
for i = 1:length(m)
    net.W{i} = SIGMA*randn(m(i),prec_layer);
    net.B{i} = SIGMA*randn(m(i),1);
    prec_layer=m(i);
end
%% CREAZIONE STRATO OUTPUT
net.W{i+1} = SIGMA*randn(c,m(length(m)));
net.B{i+1} = SIGMA*randn(c,1);
%% DEFINIZIONE STRUTTURA NET
net.d=d;
net.m=m;
net.c=c;
net.f=f;
net.g=g;
net.numLayers=length(m)+1;
end