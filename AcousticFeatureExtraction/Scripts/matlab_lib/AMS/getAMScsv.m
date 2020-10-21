
inputFolder = "/home/hmgent2/Irony-Recognition/AudioData/GatedPruned"; 

filePattern = fullfile(inputFolder, '*.wav');

theFiles = dir(filePattern);

for k = 1 : length(theFiles)
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    fprintf(1, '%s\n', baseFileName)
    base = strrep(baseFileName, 'SPPep12_', '')
    base = strrep(base, '.wav', '.csv')
    fprintf(1, '%s\n', base)
    ams = extract_AMS(fullFileName)
    writematrix(ams, base)
    % fprintf(1, 'Now reading %s\n', fullFileName);
    % Now do whatever you want with this file name,
    % such as reading it in as an image array with imread()
end

