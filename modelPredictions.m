function [classesPredictions,classCorr] = modelPredictions_lifetime(dlnet,mbq,images, feat,classes)

    classesPredictions = [];    
    classCorr = [];  
    if(numel(mbq)>0)
        while hasdata(mbq)
            [dlX1,dlX2,dlY] = next(mbq);
            
            % Make predictions using the model function.
            dlX1_temp=extractdata(dlX1);
            dlX1_temp=permute(dlX1_temp,[1 2 4 3]);
            size(dlX1_temp)
            dlX1=dlarray(dlX1_temp,"SSCB");
            dlX2=dlarray(dlX2',"CB");
            dlYPred = predict(dlnet,dlX1,dlX2);
            
            dlYPred_reg=extractdata(dlYPred)
            [maxv YPredBatch] = max(dlYPred_reg);
            %%Patch
            for i=1:numel(dlY)
                diff=abs(dlYPred_reg(1,i)-dlYPred_reg(2,i));
                if(diff<.1)
                    if(YPredBatch(i)==dlY(i))
                        continue;
                    else
                        diff
                        YPredBatch(i)
                        YPredBatch(i)=mod(YPredBatch(i)+1,2);
                        if(YPredBatch(i)==0)
                            YPredBatch(i)=2;
                        end
                        YPredBatch(i)
                    end
                end
            end
            % Determine predicted classes.
%             
            classesPredictions = [classesPredictions YPredBatch];
            % Compare predicted and true classes.
            classCorr = [classCorr YPredBatch == dlY']
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

