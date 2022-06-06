function [classesPredictions,classCorr] = modelPredictions_lifetime(dlnet,mbq,images, feat,classes)

    classesPredictions = [];    
    classCorr = [];  
    if(numel(mbq)>0)
        %dlnet = resetState(dlnet);
        while hasdata(mbq)
            
            [dlX1,dlX2,dlY] = next(mbq);
            
            % Make predictions using the model function.
            dlX1_temp=extractdata(dlX1);
           
            dlX1=dlarray(dlX1_temp,"CTB");
            dlX2=dlarray(dlX2',"CB");
%             size(dlX1)
%             size(dlX2)
            dlYPred = predict(dlnet,dlX1,dlX2);
            
            dlYPred_reg=extractdata(dlYPred);
            [maxv YPredBatch] = max(dlYPred_reg);
      
            % Determine predicted classes.
%             
            classesPredictions = [classesPredictions YPredBatch];
            % Compare predicted and true classes.
            classCorr = [classCorr YPredBatch == dlY'];
        end
    end
    if(numel(mbq)==0)
        display('No minibatching')
        dlYPred = predict(dlnet,images,feat);
%         dlYPred= dlarray(double(dlYPred),"CB");
%         classes=dlarray(double(classes),"CB");
        dlYPred_reg=extractdata(dlYPred)
        
        % Determine predicted classes.
        [maxv YPredBatch] = max(dlYPred_reg);
        classesPredictions = [classesPredictions YPredBatch];
                
        % Compare predicted and true classes.
        %Y = onehotdecode(dlY,classes,1);
        classCorr = [classCorr YPredBatch == classes];
    end
end

