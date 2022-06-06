function [accuracyTrain, accuracyTest, prec_test, rec_test, JI_test, AUC_test]=trainingDataGlia_lifetimefeatures_minibatch(LF_train, LF_test, All_feat_train, All_feat_test, YlabelSet, YTestingSetLabel,mbatch)
%%param gen_features is 1 if randomly drawing data and generating features
%%is 0 if old features used

 g = gpuDevice(1);
 reset(gpuDevice(1));
additionalMetrics=0;
numClasses=2;
if(nargin<7)
    miniBatchSize=27;%33 for sagardata;
else
    miniBatchSize=mbatch;
end
numEpochs=20;
InPutimageSize=200;

if(nargin<1)%%no parameters passed
    load data_feat_all.mat
    N=size(All_feat,2);
    numFeatures=size(All_feat,1);
    randloc=randperm(N);
    
    trainind=randloc(1:numTrain);
    testind=randloc(numTrain+1:numTrain+numTest);
    
    XtrainSet=XSet(:,:,:,trainind);
    XtestSet=XSet(:,:,:,testind);
    All_feat_train=All_feat(:,trainind);
    All_feat_test=All_feat(:,testind);
    YlabelSet=Ylabel(trainind);
    YTestingSetLabel=Ylabel(testind);
    
    load data_lifetime.mat
    LF_train=LF(:,:,trainind);
    LF_test=LF(:,:,testind);
end
numFeatures=size(All_feat_train,1);
numFeatures_lifetime=size(LF_train,1);
num_train=size(All_feat_train,2)
num_test=size(All_feat_test,2);


Meanval_features=mean(All_feat_train');
indclass1=find(YlabelSet==1);
indclass2=find(YlabelSet==2);

Meanval_features1=mean(All_feat_train(:,indclass1),2)';
Meanval_features2=mean(All_feat_train(:,indclass2),2)';
[Meanval_features1; Meanval_features2];

filterSize=3
numFilters=16

layers = [
    sequenceInputLayer(numFeatures_lifetime,'Name','sequence')
    bilstmLayer(200,'OutputMode','last','Name','bilstm')%2000 earlier
    dropoutLayer(0.5,'Name','drop')
    concatenationLayer(1,2,'Name','concat')
    fullyConnectedLayer(numClasses,'Name','fc')
    softmaxLayer('Name','softmax')
    %classificationLayer('Name','classification')];   
    
     
    
%     %classificationLayer('Name','classification')
    ];

lgraph = layerGraph(layers);
featInput = [ ...
    featureInputLayer(numFeatures,'Name','features','Normalization', 'zerocenter','Mean',Meanval_features)
    
    %feature Cconvolution Layer
    FeatConvLayer("FeatConv", numFeatures)
    fullyConnectedLayer(50,'Name','fc3')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu2')
    ];
%featInput = featureInputLayer(numFeatures,'Name','features','Normalization', 'zerocenter','Mean',Meanval_features);
lgraph = addLayers(lgraph, featInput);
% 
%lgraph = connectLayers(lgraph, 'features', 'concat/in2');
lgraph = connectLayers(lgraph, 'relu2', 'concat/in2');
 

executionEnvironment = "gpu";

iteration=1;
learnRate = 0.001;%%.0005 worked well in some cases

decay = 0.01;
momentum = 0.9;
velocity=[];
start=tic;

dlX1 = arrayDatastore(LF_train,'IterationDimension',3);
dlX2 = arrayDatastore(double(All_feat_train)','IterationDimension',1);
dlX3 = arrayDatastore(double(YlabelSet'));

dsTrain=combine(dlX1,dlX2,dlX3);
%mbq = minibatchqueue(dsTrain,2,'MiniBatchSize',5,'MiniBatchFcn', @preprocessMiniBatch,'MiniBatchFormat',{'CB','CB'});
mbq = minibatchqueue(dsTrain,3,'MiniBatchSize',miniBatchSize);

dlnet = dlnetwork(lgraph);


for epoch = 1:numEpochs
    
    shuffle(mbq)
    iteration=1;
    avg_misclassloss=0;
    % Loop over mini-batches.
    %dlnet = resetState(dlnet);
    while hasdata(mbq)
        [epoch iteration]
       
        [dlX1, dlX2, dlX3] = next(mbq);
        
        dlX1_temp=extractdata(dlX1);
%         dlX1_temp=permute(dlX1_temp,[1 2 4 3]);
        
        dlX1=dlarray(dlX1_temp,"CTB");
        dlX2=dlarray(dlX2',"CB");
        dlX3_new=dlarray([abs(dlX3'-2); dlX3'-1],"CB");
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlX1=gpuArray(dlX1);
            dlX2 = gpuArray(dlX2);
            dlX3_new = gpuArray(dlX3_new);
        end
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelGradients function and update the network state.
        [gradients,state,loss, misclass_loss] = dlfeval(@modelGradients,dlnet, dlX1, dlX2,dlX3_new);
        if(epoch<=numEpochs)
            avg_misclassloss=avg_misclassloss+misclass_loss;
        end
        dlnet.State = state;
        
        % Update the network parameters using the SGDM optimizer.
        [dlnet, velocity] = sgdmupdate(dlnet, gradients, velocity, learnRate, momentum);
        iteration = iteration + 1;
    end   
    avg_misclassloss= avg_misclassloss/(iteration - 1);
    if(avg_misclassloss==0)
        break
    end
end
accuracyTrain=gather(1-avg_misclassloss);
display(['Training acc', num2str(1-avg_misclassloss)]);
% if(additionalMetrics)
%      [fv, prec_train, rec_train, acc, JI_train, dice, MI, NVI RI, ARI, AUC_train]=prec_rec(logical(Ypred-1),logical(YTestingSetLabel-1));
% else
%     
%     prec_train=[];
%     rec_train=[];
%     JI_train=[];
%     AUC_train=[];
% end
dTX1 = arrayDatastore(double(LF_test),'IterationDimension',3);
dTX2 = arrayDatastore(double(All_feat_test'),'IterationDimension',1);
dTX3 = arrayDatastore(double(YTestingSetLabel'));
dsTest = combine(dTX1,dTX2,dTX3);
classes=double(YTestingSetLabel);


mbqTest= minibatchqueue(dsTest,3,'MiniBatchSize',miniBatchSize);

[YPred,predCorr] = modelPredictions_lifetimefeat(dlnet,mbqTest,[],classes); 
classes;
% YTestingSetLabel=YTestingSetLabel(1:end-7) %for jdata only, comment for sagar
% YPred=YPred(1:end-7)%for jdata only, comment for sagar
YPred=gather(YPred);
accuracyTest = gather(sum(YTestingSetLabel == YPred)/numel(YPred))
if(additionalMetrics)
     [fv, prec_test, rec_test, acc, JI_test, dice, MI, NVI RI, ARI, AUC_test]=prec_rec(logical(YPred-1)',logical(YTestingSetLabel-1)');
else
    
    prec_test=[];
    rec_test=[];
    JI_test=[];
    AUC_test=[];
end
%C = confusionmat(YTestingSetLabel,YPred)


