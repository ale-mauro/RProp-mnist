function e=derivCrossEntropyMC(Y,T)
e = -sum(T ./ Y , 2);
end