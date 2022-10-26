function y=simNet(net,x)
%y=simNet(net,x)
%Simula il comportamento della rete 'net' con input 'x'

%% CONTROLLO DIMENSIONI DI INPUT
assert(size(x,1)==net.d,...
    "Il numero di righe dell'input ("+size(x,1)+") deve essere uguale a d ("+net.d+")")

%% COMPORTAMENTO STRATI INTERNI (calcolo a1...a_h-1 e z1..z_h-1)
z=x;
for i=1:(net.numLayers)-1
    a = net.W{i}*z;
    a = a + net.B{i};
    z = net.f{i}(a);
end
%% COMPORTAMENTO STRATO OUTPUT (calcolo a_c e y)
a = net.W{i+1}*z + net.B{i+1};
y = net.g(a);
end