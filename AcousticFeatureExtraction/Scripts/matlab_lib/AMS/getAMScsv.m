
winSize = 0.01

inputFolder = "/home/hmgent2/Irony-Recognition/AudioData/GatedAll"; 

filePattern = fullfile(inputFolder, '*.wav');

theFiles = dir(filePattern);

for k = 1 : length(theFiles)
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    fprintf(1, '%s\n', baseFileName)
    base = strrep(baseFileName, 'SPPep12_', '')
    base = strrep(base, '.wav', '.csv')
    fprintf(1, '%s\n', base)
    ams = extract_AMS(fullFileName, winSize)
    writematrix(ams, base)
end


