% Housekeeping
clearvars
close all
clc

% Be forewarned, I made this code my own. This means my comments are not
% strictly professional. They are still informative and should get the
% point across just fine. But I also make jokes, because it makes things
% fun and reading the code less tedious.

%!!! CHECK BEFORE EACH SCRIPT EXECUTION !!!

% I've tried to make it so that you don't have to fiddle around in the
% simulink file. You can change everything of interest here. Having the
% simulink file open to view the scopes as it runs might still be worth it.

file.folder = "H"; % where the data files are
data.dt = NaN;

% Snapshot data
data.results = readdictionary("S/run_1.json"); % results = (keys: string), (values: 1x1 cell array: 1024x1 cell)

data.measured_vars = string(cell2mat(data.results{"measurements"})'); % extracting the names of each measurement, while also handling funky datatype manipulations (dictionary entries are all 1x1 cell arrays to make the dictionary creation work, but the entry is another cell array of the data we want)
data.num_points = numel(cell2mat(data.results{data.measured_vars(1)})); % checking how many samples there are, arbitrarily using the first measurement set (every set should contain the same number of elements)

% getting the measurement time interval to slice out the window of the
% chosen operating condition (ignoring the NOC portion, if there is a
% disturbance)
if ismember("t", data.measured_vars) && isnan(data.dt)
    data.t = cell2mat(data.results{"t"});
    data.dt = data.t(2) - data.t(1);
else
    error("Expected time data not found. Ensure time was saved with the label 't', or provide the time interval")
end
data.measured_vars(data.measured_vars == "t") = []; % the measurement times are not being considered for PCA, removing it here saves effort later
data.num_vars = numel(data.measured_vars); % the number of variables that will be in the matrix we perform PCA on

data.num_ignore = data.results{"fault time"} ./ data.dt; % the number of measurements that will be ignored (corresponding to the period of NOC before the disturbance)
data.set = zeros(data.num_points, data.num_vars); % initialising the matrix that will have all the data to perform PCA

% putting all the simulation data into one matrix
for ii = 1:data.num_vars
    data.set(:, ii) = cell2mat(data.results{data.measured_vars(ii)});
end

[data.scaled, data.mean, data.std] = normalize(data.set); % scaling the data
[data.S.coeff, data.S.score, ~, ~, explained] = pca(data.scaled(data.num_ignore+1:end, :)); % performing PCA, saving convenient stats

% choosing the number of principle components based on how many are needed
% to explain 95 % of the variance
for ii = 1:numel(explained)
    if sum(explained(1:ii)) > 95
        k = ii;
        break
    end
end
L = data.S.coeff(:, 1:k); % principal subspace of the snapshot

% Historical data
jj = 1;
while isfile(file.folder+"/run_"+jj+".json")
    jj = jj + 1;
end
file.number = jj - 1; % the number of data files

data.S_PCA = zeros(1, file.number);

for jj = 1:file.number
    data.results = readdictionary(file.folder+"/run_"+jj+".json"); % results = (keys: string), (values: 1x1 cell array: 1024x1 cell)
    
    data.set = zeros(data.num_points, data.num_vars); % initialising the matrix that will have all the data to perform PCA
    % putting all the simulation data into one matrix
    for ii = 1:data.num_vars
        data.set(:, ii) = cell2mat(data.results{data.measured_vars(ii)});
    end

    [data.scaled] = normalize(data.set, 'center', data.mean, 'scale', data.std); % scaling the data
    [data.H.coeff] = pca(data.scaled(data.num_ignore+1:end, :),'NumComponents', k); % performing PCA, saving convenient stats
    M = data.H.coeff(:, 1:k);

   
    data.S_PCA(jj) = trace(L'*M*M'*L) ./ k;
end

sum(data.S_PCA > 0.965)
