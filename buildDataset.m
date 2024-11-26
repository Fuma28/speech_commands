downloadFolder = matlab.internal.examples.downloadSupportFile("audio","google_speech.zip");
dataFolder = 'speech_commands';
unzip(downloadFolder,dataFolder)
dataset = fullfile(dataFolder,"google_speech");
augmentDataset(dataset); % builds the background samples