% load the trained net and the classes
load trainedNet.mat
load classes.mat
%%
% we define the parameters for the classification and the listening
classificationRate = 10;
frameAgreementThreshold = 0.5;
probabilityThreshold = 0.7;
fs = 16e3;
adr = audioDeviceReader(fs,floor(fs/classificationRate));
audioBuffer = dsp.AsyncBuffer(fs);
newSamplesPerUpdate = floor(fs/classificationRate);
countThreshold = round(frameAgreementThreshold*classificationRate) + 1;
YBuffer = repmat(categorical("background"),classificationRate,1);
scoreBuffer = zeros(numel(classes),classificationRate,"single");
%%
% we set up the parameters for the feature extraction
segmentDuration = 1;
frameDuration = 0.025;
hopDuration = 0.010;
segmentSamples = round(segmentDuration*fs);
frameSamples = round(frameDuration*fs);
hopSamples = round(hopDuration*fs);
overlapSamples = frameSamples - hopSamples;

afe = audioFeatureExtractor(SampleRate=fs, Window=hann(frameSamples,"periodic"), OverlapLength=overlapSamples, melSpectrum=true);
setExtractorParameters(afe,"melSpectrum",WindowNormalization=false);
%%
% we listen to the input, we extract its features and predict its class
disp("Speak now!")
while true
    audioIn = adr();
    write(audioBuffer,audioIn);

    y = read(audioBuffer,fs,fs - newSamplesPerUpdate);
    y = resize(y,segmentSamples,Side="both");
    features = extract(afe,y);
    spec = log10(features + 1e-6);
    
    score = predict(trainedNet,spec);
    YPredicted = scores2label(score, classes, 2);
    YBuffer = [YBuffer(2:end);YPredicted];
    scoreBuffer = [scoreBuffer(:,2:end),score(:)];
    
    [YMode,count] = mode(YBuffer);
    maxProb = max(scoreBuffer(classes == YMode,:));
    if YMode ~= "background" && count > countThreshold && maxProb > probabilityThreshold
        disp(YMode);
    end
end