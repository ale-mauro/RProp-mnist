function [A,Z,y]=forwardStep(net,x)
%function [A,Z,y]=forwardStep(net,x)
%Simula il comportamento della rete 'net' con input 'x' restituendo
%l'output della rete (y), gli input (A) e gli output (Z) degli strati

%% CONTROLLO DIMENSIONI DI INPUT
assert(size(x,1)==net.d,...
    "Il numero di righe dell'input ("+size(x,1)+") deve essere uguale a d ("+net.d+")")

len=net.numLayers;
A=cell(1,len);
Z=cell(1,len);
%% COMPORTAMENTO STRATI INTERNI (calcolo a1...a_c-1 e z2...z_c-1)
z_prev=x;
for i=1:length(net.m)
    A{i} = net.W{i}*z_prev + net.B{i};
    Z{i} = net.f{i}(A{i});
    z_prev=Z{i};
end
%% COMPORTAMENTO STRATO OUTPUT (calcolo a_c e y)
A{i+1}= net.W{i+1}*Z{i} + net.B{i+1};
y = net.g(A{i+1});
Z{i+1}=y;
end