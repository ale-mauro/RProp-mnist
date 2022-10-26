function net=discesaDelGradienteStandard(net,eta, gradiente)
%function net=discesaDelGradienteStandard(net,eta, derivW, derivBias)
%Discesa del gradiente classica
for i=1:net.numLayers
    net.W{i} = net.W{i} -eta*gradiente.W{i};
    net.B{i} = net.B{i} -eta*gradiente.B{i};
end
end