function cross_val_main()
%load C:\Users\lopa\Dropbox\microglia_morpho/code_readdata/sagardata67feat.mat %%data with all features and images and label from conservative segmentation of rst(total 989)
load /home/mukherjl/Dropbox/microglia_morpho/code_readdata/sagardata67feat_tau.mat
%load /home/mukherjl/Dropbox/microglia_morpho/code_readdata/jdata67_vetted.mat
additionalMetrics=0;
k=5;
n=size(All_feat,2)
cv = cvpartition(n,'KFold',k);
Acc=[];
prec=[];
rec=[];
JI=[];
AUC=[];

%%for Jdata, trainsetsize=405 and testsetsize=108 to create minibatches of
%%27
mbatch_size=27;
jdata=0;
for i=1:k
    trainind=find(cv.training(i))';
    testind=find(cv.test(i))';
    numtrain=numel(trainind);
    numtest=numel(testind);

    if(numel(LF)>1)
        LF_train=LF(:,:,trainind);
        LF_test=LF(:,:,testind);
    end
    All_feat_train=All_feat(:,trainind);
    All_feat_test=All_feat(:,testind);

    YlabelSet=Ylabel(trainind);
    YTestingSetLabel=Ylabel(testind);

    %%for Jdataonly to make all splits same size of 27 in train and test
    if(jdata)
        if(numel(trainind)<405)
            LF_train=cat(3,LF_train,LF(:,:,trainind(1:405-numtrain)));
        end
        LF_test=cat(3,LF_test,LF(:,:,testind(1:108-numtest)));
        if(numel(trainind)<405)
            All_feat_train=cat(2,All_feat_train,All_feat(:,trainind(405-numtrain)));
        end
        All_feat_test=cat(2,All_feat_test,All_feat(:,testind(1:108-numtest)));
        if(numel(trainind)<405)
            YlabelSet=cat(2,YlabelSet, Ylabel(trainind(405-numtrain)));
        end
        YTestingSetLabel=cat(2,YTestingSetLabel, Ylabel(testind(1:108-numtest)));
    end

    %[accuracyTrain, accuracyTest]=trainingDataGlia_imgfeatures_minibatch(0,XtrainSet, XtestSet, All_feat_train, All_feat_test, YlabelSet, YTestingSetLabel);
    if(additionalMetrics)
        [accuracyTrain, accuracyTest, prec_test, rec_test, JI_test, AUC_test]=trainingDataGlia_lifetimefeatures_minibatch(LF_train,LF_test, All_feat_train, All_feat_test, YlabelSet, YTestingSetLabel,mbatch_size);
        Acc=[Acc; [i accuracyTrain, accuracyTest]];
        
        prec=[prec; [i prec_test]];
        rec=[rec; [i rec_test]];
        JI=[JI; [i JI_test]];
        AUC=[AUC; [i  AUC_test]];
    else
       %[accuracyTrain, accuracyTest]=trainingDataGlia_lifetimgfeatures_minibatch(LF_train,LF_test, All_feat_train, All_feat_test, YlabelSet, YTestingSetLabel,mbatch_size);
       [accuracyTrain, accuracyTest]=trainingDataGlia_features_minibatch(All_feat_train, All_feat_test, YlabelSet, YTestingSetLabel,mbatch_size);
       Acc=[Acc; [i accuracyTrain, accuracyTest]];
    end
    
    Acc
    save result_5fold_cv_feat_sagardata_tau.mat Acc %prec rec JI AUC
end
mean(Acc,1)