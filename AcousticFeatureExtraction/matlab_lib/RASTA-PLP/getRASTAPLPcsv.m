inputFolder = "/home/hmgent2/Irony-Recognition/AudioData/GatedPruned3";

filePattern = fullfile(inputFolder, '*.wav');

theFiles = dir(filePattern);

for k = 1 : length(theFiles)
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    fprintf(1, '%s\n', baseFileName)
    base = strrep(baseFileName, '.wav', '.csv')
    fprintf(1, '%s\n', base)
    plp = rastaplp(fullFileName)
    writematrix(plp, base)
end
