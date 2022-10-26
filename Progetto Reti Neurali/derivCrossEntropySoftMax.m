function e=derivCrossEntropySoftMax(Y,T)
Y=softMax(Y);%applicazione Softmax sull'uscita Y dei neuroni

e=Y-T;
end