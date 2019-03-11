tic
%%
net = alexnet;
layers = net.Layers;
c = net.Layers(end).ClassNames;
%%
rootFolder = 'imAcqtrain';
categories = {'LH_I_P3','LH_M_P1','RH_I_P2','RH_M_P1'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');

imds = splitEachLabel(imds, 50, 'randomize')
imds.ReadFcn = @readFunctionTrain;
%%
layers = layers(1:end-3);
layers(end+1) = fullyConnectedLayer(64, 'Name', 'fc8_1');
layers(end+1) = reluLayer;
layers(end+1) = fullyConnectedLayer(4, 'Name', 'fc8_2');
layers(end+1) = softmaxLayer;
layers(end+1) = classificationLayer()
%%
layers(end-2).WeightLearnRateFactor = 10;
layers(end-2).WeightL2Factor = 1;
layers(end-2).BiasLearnRateFactor = 20;
layers(end-2).BiasL2Factor = 0;
%%
opts = trainingOptions('sgdm', ...
    'LearnRateSchedule', 'none',...
    'InitialLearnRate', .0001,... 
    'MaxEpochs', 8, ...
    'MiniBatchSize', 20);
%%
[convnet,info] = trainNetwork(imds, layers, opts);

