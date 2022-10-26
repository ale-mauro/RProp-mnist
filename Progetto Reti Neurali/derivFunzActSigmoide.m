function y=derivFunzActSigmoide(x)
    z=sigmoide(x);
    y=z .* (1-z);
end