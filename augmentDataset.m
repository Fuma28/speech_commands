function augmentDataset(datasetloc)
adsBkg = audioDatastore(fullfile(datasetloc,"background"));
fs = 16e3; % Known sample rate of the data set
segmentDuration = 1;
segmentSamples = round(segmentDuration*fs);

volumeRange = log10([1e-4,1]);

numBkgSegments = 4000;
numBkgFiles = numel(adsBkg.Files);
numSegmentsPerFile = floor(numBkgSegments/numBkgFiles);

fpTrain = fullfile(datasetloc,"train","background");
fpValidation = fullfile(datasetloc,"validation","background");

if ~exist(fpTrain)

    % Create directories
    mkdir(fpTrain)
    mkdir(fpValidation)

    for backgroundFileIndex = 1:numel(adsBkg.Files)
        [bkgFile,fileInfo] = read(adsBkg);
        [~,fn] = fileparts(fileInfo.FileName);

        % Determine starting index of each segment
        segmentStart = randi(size(bkgFile,1)-segmentSamples,numSegmentsPerFile,1);

        % Determine gain of each clip
        gain = 10.^((volumeRange(2)-volumeRange(1))*rand(numSegmentsPerFile,1) + volumeRange(1));

        for segmentIdx = 1:numSegmentsPerFile

            % Isolate the randomly chosen segment of data.
            bkgSegment = bkgFile(segmentStart(segmentIdx):segmentStart(segmentIdx)+segmentSamples-1);

            % Scale the segment by the specified gain.
            bkgSegment = bkgSegment*gain(segmentIdx);

            % Clip the audio between -1 and 1.
            bkgSegment = max(min(bkgSegment,1),-1);

            % Create a file name.
            afn = fn + "_segment" + segmentIdx + ".wav";

            % Randomly assign background segment to either the train or
            % validation set.
            if rand > 0.85 % Assign 15% to validation
                dirToWriteTo = fpValidation;
            else % Assign 85% to train set.
                dirToWriteTo = fpTrain;
            end

            % Write the audio to the file location.
            ffn = fullfile(dirToWriteTo,afn);
            audiowrite(ffn,bkgSegment,fs)

        end

        % Print progress
        fprintf('Progress = %d (%%)\n',round(100*progress(adsBkg)))

    end
end
end