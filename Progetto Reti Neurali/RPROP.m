function [net,Delta,oldMod,gradiente]=RPROP(net,method,eta,eta_p,eta_n,gradiente,oldGrad,Delta,oldMod,err,epoch)
%function [net,Delta,oldMod,gradiente]=RPROP(net,method,eta,eta_p,eta_n,gradiente,oldGrad,Delta,oldMod,err,epoch)
%esegue la variante 'method' della RProp. method può avere valore
%'rprop+','rprop-','irprop+' ed 'irprop-'.
%restituisce la rete con i nuovi parametri, i Deltaij relativi ai
%parametri, i modificatori dell'iterazione precedente e il gradiente che in
%alcune varianti sarà eventualmente modificato.

method=lower(method);
if epoch==1
    net=discesaDelGradienteStandard(net,eta, gradiente);
else
    for i=1:net.numLayers
        %% Calcolo prodotto tra gradiente attuale e gradiente precedente
        gg = gradiente.W{i}.*oldGrad.W{i};
        ggB = gradiente.B{i}.*oldGrad.B{i};
        
        %% Calcolo Delta_ij
        Delta.W{i} =...
            min(Delta.W{i}*eta_p,Delta.max.W{i}).*(gg>0) +...
            max(Delta.W{i}*eta_n,Delta.min.W{i}).*(gg<0) +...
            Delta.W{i}.*(gg==0);
        %Delta.W{i}.*(isnan(gg));
        
        Delta.B{i} =...
            min(Delta.B{i}*eta_p,Delta.max.B{i}).*(ggB>0) +...
            max(Delta.B{i}*eta_n,Delta.min.B{i}).*(ggB<0) +...
            Delta.B{i}.*(ggB==0);
        %Delta.B{i}.*(isnan(ggB));
        
        %% Calcolo modificatoriW e B in base alla variante di RPROP
        switch(method)
            case 'rprop-'
                modificatoreW=-sign(gradiente.W{i}).*Delta.W{i};
                modificatoreB=-sign(gradiente.B{i}).*Delta.B{i};
                
            case 'rprop+'
                modificatoreW=...
                    (-sign(gradiente.W{i}).*Delta.W{i}).*(gg>=0)...
                    -oldMod.W{i}.*(gg<0);
                
                modificatoreB=...
                    (-sign(gradiente.B{i}).*Delta.B{i}).*(ggB>=0)...
                    -oldMod.B{i}.*(ggB<0);
                
                oldMod.W{i}=modificatoreW;
                oldMod.B{i}=modificatoreB;
                
                %Derivata a 0 per influenzare la prossima iterazione
                gradiente.W{i}=gradiente.W{i}.*(gg>=0);
                gradiente.B{i}=gradiente.B{i}.*(ggB>=0);
                
            case 'irprop+'
                E=err(epoch);
                oldErr=err(epoch-1); %err(epoch-1) esiste perchè per epoch=1 si effettua discesa del gradiente standard
                
                modificatoreW=...
                    (-sign(gradiente.W{i}).*Delta.W{i}).*(gg>=0)...
                    -oldMod.W{i}.*(gg<0)*(E>oldErr);
                %+(-sign(gradiente.W{i}).*Delta.W{i}).*(gg<0)*(E<=oldErr)...
                
                modificatoreB=...
                    (-sign(gradiente.B{i}).*Delta.B{i}).*(ggB>=0)...
                    -oldMod.B{i}.*(ggB<0)*(E>oldErr);
                %+(-sign(gradiente.B{i}).*Delta.B{i}).*(ggB<0)*(E<=oldErr)...
                
                oldMod.W{i}=modificatoreW;
                oldMod.B{i}=modificatoreB;
                
                gradiente.W{i}=gradiente.W{i}.*(gg>=0);
                gradiente.B{i}=gradiente.B{i}.*(ggB>=0);
                
            case 'irprop-'
                gradiente.W{i}=gradiente.W{i}.*(gg>=0);
                gradiente.B{i}=gradiente.B{i}.*(ggB>=0);
                
                modificatoreW=-sign(gradiente.W{i}).*Delta.W{i};
                modificatoreB=-sign(gradiente.B{i}).*Delta.B{i};
                
            otherwise
                error("la modalità può essere rprop-, rprop+, irprop-, irprop+");
        end
        
        %% AGGIORNAMENTO PESI
        net.W{i} = net.W{i} + modificatoreW;
        net.B{i} = net.B{i} + modificatoreB;
    end
end
end