function acc=accuracy(Y,T)
%function acc=accuracy(Y,T)
%considera come regola di decisione quella di scegliere la classe con
%probabilità massima.

[~,ind_Y]=max(Y);
[~,ind_T]=max(T);

correct=sum(ind_Y==ind_T);
acc=correct/length(ind_Y);
end