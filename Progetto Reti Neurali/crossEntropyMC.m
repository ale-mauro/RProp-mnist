function e=crossEntropyMC(Y,T)
% function e=crossEntropyMC(Y,T)
% e=-sum( sum(T .* log2(Y),1));
e=-sum( sum(T .* log2(Y),1));
end