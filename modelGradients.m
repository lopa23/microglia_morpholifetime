function [gradients,state,loss,misclass_loss] = modelGradients(dlnet,dlX1,dlX2,Y)

[dlYPred,state] = forward(dlnet,dlX1,dlX2);
Y
dlYPred
loss = crossentropy(dlYPred,Y)
if(isnan(loss)==1)
    pause
end
misclass_loss = myloss(dlYPred,Y);
display('Crossentropy_loss Misclass_loss');
[loss misclass_loss]

gradients = dlgradient(loss,dlnet.Learnables);

end

function loss=myloss(X,Y)

[val class1]=max(X,[],1);
[val class2]=max(Y,[],1);
loss=sum(class1~=class2)./numel(class1);
end