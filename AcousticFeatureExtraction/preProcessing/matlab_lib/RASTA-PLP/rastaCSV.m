function csv = rastaCSV(inputFolder, destFolder) 
% inputFolder points to a folder full of wavs that need feature extracting
% destFolder points to an empty folder where the CSV files need to go

filePattern = fullfile(inputFolder, '*.wav');
theFiles = dir(filePattern);

for k = 1 : length(theFiles)
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    base = strrep(baseFileName, '.wav', '.csv');
    dest = strcat(destFolder, base);
    plp = rastaplp(fullFileName);
    writematrix(plp, dest);
end
