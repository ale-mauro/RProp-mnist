function z=softMax(y)
z=exp(y)./sum(exp(y),1);
end