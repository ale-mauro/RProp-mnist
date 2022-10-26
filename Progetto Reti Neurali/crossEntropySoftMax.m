function e=crossEntropySoftMax(Y,T)
%function e=crossEntropySoftMax(Y,T)


Y=softMax(Y);
e=-sum(sum(T .* log(Y),1));

end