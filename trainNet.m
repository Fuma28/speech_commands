dataFolder = 'speech_commands';
dataset = fullfile(dataFolder, 'google_speech');
commands = categorical(["up","down","left","right","stop","go"]);
background = categorical("background");
%%
% we set up the train dataset keeping only the commands we are interested
% in, the background and some other samples as unknown
ads = audioDatastore(fullfile(dataset,"train"), IncludeSubfolders=true, FileExtensions=".wav", LabelSource="foldernames");
isCommand = ismember(ads.Labels,commands);
isBackground = ismember(ads.Labels,background);
isUnknown = ~(isCommand|isBackground);
includeFraction = 0.2; % Fraction of unknowns to include.
idx = find(isUnknown);
idx = idx(randperm(numel(idx),round((1-includeFraction)*sum(isUnknown))));
isUnknown(idx) = false;
ads.Labels(isUnknown) = categorical("unknown");
adsTrain = subset(ads,isCommand|isUnknown|isBackground);
adsTrain.Labels = removecats(adsTrain.Labels);
%%
% similarly we set up the validation dataset
ads = audioDatastore(fullfile(dataset,"validation"), IncludeSubfolders=true, FileExtensions=".wav", LabelSource="foldernames");
isCommand = ismember(ads.Labels,commands);
isBackground = ismember(ads.Labels,background);
isUnknown = ~(isCommand|isBackground);
includeFraction = 0.2; % Fraction of unknowns to include.
idx = find(isUnknown);
idx = idx(randperm(numel(idx),round((1-includeFraction)*sum(isUnknown))));
isUnknown(idx) = false;
ads.Labels(isUnknown) = categorical("unknown");
adsValidation = subset(ads,isCommand|isUnknown|isBackground);
adsValidation.Labels = removecats(adsValidation.Labels);
%%
% we set up the parameters for the feature extraction
fs = 16e3; % Known sample rate of the data set.
segmentDuration = 1;
frameDuration = 0.025;
hopDuration = 0.010;
FFTLength = 512;

segmentSamples = round(segmentDuration*fs);
frameSamples = round(frameDuration*fs);
hopSamples = round(hopDuration*fs);
overlapSamples = frameSamples - hopSamples;

afe = audioFeatureExtractor(SampleRate=fs, Window=hann(frameSamples,"periodic"), OverlapLength=overlapSamples, melSpectrum=true);
setExtractorParameters(afe,"melSpectrum", WindowNormalization=false);
%%
% we extract the train set features:
%   - first we resize every sample so that it is 1 sec long
%   - than we extract the spectrum
%   - we then get the logarithm of the spectrum to put it in a
%       human-like prospective
transform1 = transform(adsTrain,@(x)resize(x, segmentSamples, Side="both"));
transform2 = transform(transform1,@(x)extract(afe,x));
transform3 = transform(transform2,@(x){log10(x+1e-6)});
dataTrain = readall(transform3);
dataTrain = cat(4,dataTrain{:});
[numTimeSegments,numBins,~,numSamples] = size(dataTrain);
%%
% similarly we extract the validation set features
transform1 = transform(adsValidation,@(x)resize(x, segmentSamples, Side="both"));
transform2 = transform(transform1,@(x)extract(afe,x));
transform3 = transform(transform2,@(x){log10(x+1e-6)});
dataValidation = readall(transform3);
dataValidation = cat(4,dataValidation{:});
%%
% we set up the cnn
labelsTrain = adsTrain.Labels;
labelsValidation = adsValidation.Labels;
classes = categories(labelsTrain);
classWeights = 1./countcats(labelsTrain);
classWeights = classWeights'/mean(classWeights);
timePoolSize = ceil(numTimeSegments/8); %(we divide for 8 because we apply 3 times max pooling with stride 2, so 2^3 =8)
dropoutProb = 0.2;
numF = 12;

layers = [
    imageInputLayer([numTimeSegments,afe.FeatureVectorLength])
    
    convolution2dLayer(3,numF,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3,Stride=2,Padding="same")
    
    convolution2dLayer(3,2*numF,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3,Stride=2,Padding="same")
    
    convolution2dLayer(3,4*numF,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3,Stride=2,Padding="same")
    
    convolution2dLayer(3,4*numF,Padding="same")
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(3,4*numF,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([timePoolSize,1])
    dropoutLayer(dropoutProb)

    fullyConnectedLayer(numel(classes))
    softmaxLayer];
%%
% we define the training options
miniBatchSize = 128;
validationFrequency = floor(numel(labelsTrain)/miniBatchSize);
options = trainingOptions("adam", ...
    InitialLearnRate=3e-4, ...
    MaxEpochs=10, ...
    MiniBatchSize=miniBatchSize, ...
    Shuffle="every-epoch", ...
    Plots="training-progress", ...
    Verbose=false, ...
    ValidationData={dataValidation,labelsValidation}, ...
    ValidationFrequency=validationFrequency, ...
    Metrics="accuracy");
%%
%we train the net
trainedNet = trainnet(dataTrain, labelsTrain, layers, @(Y,T)crossentropy(Y,T,classWeights(:),WeightsFormat="C"),options);
%%
% display accuracy and loss on validation and test data
scores = minibatchpredict(trainedNet,dataValidation);
predictedValidation = scores2label(scores,classes,"auto");
validationError = mean(predictedValidation ~= labelsValidation);
scores = minibatchpredict(trainedNet,dataTrain);
predictedTrain = scores2label(scores,classes,"auto");
trainError = mean(predictedTrain ~= labelsTrain);

disp(["Training error: " + trainError*100 + " %";"Validation error: " + validationError*100 + " %"])
%%
% display confusion matrix on validation data
figure(Units="normalized",Position=[0.2,0.2,0.5,0.5]);
cm = confusionchart(labelsValidation,predictedValidation, ...
    Title="Confusion Matrix for Validation Data", ...
  ColumnSummary="column-normalized",RowSummary="row-normalized");
sortClasses(cm,[commands,"unknown","background"])
%%
% we save the trained net and the classes
save("trainedNet.mat", 'trainedNet')
save("classes.mat", 'classes' )