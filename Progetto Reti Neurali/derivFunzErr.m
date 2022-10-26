function z=derivFunzErr(funzErr,y,t)
%function z=derivFunzErr(funzErr,y,t)
%Restituisce la derivata della funzione di errore passata per parametro. Le
%funzioni possono essere: sumOfSquares, crossEntropyMC e
%crossEntropySoftMax.
assert(isequal(funzErr,@sumOfSquares)||isequal(funzErr,@crossEntropyMC) || isequal(funzErr,@crossEntropySoftMax),...
    "La funzione di attivazione può essere @sumOfSquares, @crossEntropyMC oppure @crossEntropySoftMax")

if isequal(funzErr,@sumOfSquares)
    z=derivFunzErrSumOfSquares(y,t);
end

if isequal(funzErr,@crossEntropyMC)
    z=derivCrossEntropyMC(y,t);
end

if isequal(funzErr,@crossEntropySoftMax)
    z=derivCrossEntropySoftMax(y,t);
end

end