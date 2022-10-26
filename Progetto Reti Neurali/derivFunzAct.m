function y=derivFunzAct(funzAct,x)
%function y=derivFunzAct(funzAct,x)
%Restituisce la derivata della funzione di attivazione passata per parametro. Le
%funzioni possono essere: @sigmoide e @identity.
assert(isequal(funzAct,@sigmoide)||isequal(funzAct,@identity),...
    "La funzione di attivazione può essere @sigmoide oppure @identity")

if isequal(funzAct,@sigmoide)
    y=derivFunzActSigmoide(x);
end

if isequal(funzAct,@identity)
    y=derivFunzActIdentity(x);
end
end