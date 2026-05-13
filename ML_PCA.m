%% Housekeeping
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

snapshot = "O2"
file.folder = "H"; % where the data files are
data.dt = NaN; % set the time difference manually (only needed if simulation time data not saved)

%% Snapshot data
data.results = readdictionary("S/"+snapshot+".json"); % results = (keys: string), (values: 1x1 cell array: 1024x1 cell)
data.S.fault = string(char(cell2mat(data.results{"fault"})')); % saving the fault ID for the snapshot data

data.measured_vars = string(cell2mat(data.results{"measurements"})'); % extracting the names of each measurement, while also handling funky datatype manipulations (dictionary entries are all 1x1 cell arrays to make the dictionary creation work, but the entry is another cell array of the data we want)
data.num_points = numel(cell2mat(data.results{data.measured_vars(1)})); % checking how many samples there are, arbitrarily using the first measurement set (every set should contain the same number of elements)

% Getting the measurement time interval to slice out the window of the
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

[data.scaled, data.mean, data.std] = normalize(data.set(data.num_ignore+1:end, :)); % scaling the data
[data.S.coeff, data.S.score, ~, ~, data.explained] = pca(data.scaled, 'Economy', false, 'Centered', 'off'); % performing PCA, saving convenient stats

% choosing the number of principle components based on how many are needed
% to explain 95 % of the variance
for ii = 1:numel(data.explained)
    if sum(data.explained(1:ii)) > 95
        k_chosen = ii;
        break
    end
end
L = data.S.coeff(:, 1:k_chosen); % principal subspace of the snapshot
%% Historical data
file.number = 1;
while isfile(file.folder+"/run_"+file.number+".json")
    file.number = file.number + 1;
end
file.number = file.number - 1; % the number of data files

% Initialising arrays to store the similarity factor and the fault ID
data.S_PCA = zeros(1, file.number);
data.H.faults = strings(1,file.number);

for jj = 1:file.number
    data.results = readdictionary(file.folder+"/run_"+jj+".json"); % results = (keys: string), (values: 1x1 cell array: 1024x1 cell)
    data.H.faults(jj) = string(char(cell2mat(data.results{"fault"})')); % saving the fault ID
    data.set = zeros(data.num_points, data.num_vars); % initialising the matrix that will have all the data to perform PCA
    % putting all the simulation data into one matrix
    for ii = 1:data.num_vars
        data.set(:, ii) = cell2mat(data.results{data.measured_vars(ii)});
    end

    [data.scaled] = normalize(data.set(data.num_ignore+1:end, :), 'center', data.mean, 'scale', data.std); % scaling the data
    [data.H.coeff] = pca(data.scaled, 'Economy', false, 'Centered', 'off'); % performing PCA, saving convenient stats
    [U, S, vt] = svd(data.scaled, 'econ');
    M = data.H.coeff(:, 1:k_chosen);
   
    data.S_PCA(jj) = trace(L'*M*M'*L) ./ k_chosen;
end
%% Similarity stats
Np = sum(data.S_PCA > 0.965); % candidate pool size (number of data sets classified as similar)
N1 = sum((data.S_PCA > 0.965) & (data.H.faults == data.S.fault)); % candidate pool data sets that are actually similar
N2 = sum((data.S_PCA > 0.965) & (data.H.faults ~= data.S.fault)); % candidate pool data sets that are not similar (N2 = Np - N1)
NDB = sum(data.H.faults == data.S.fault); % historical windows that are actually similar

p = N1 ./ Np .* 100;
n = N1 ./ NDB .* 100;
nmax = min(Np./NDB, 1) .* 100;
e = ( p + n ) ./ 2;

rows2vars(table(Np, N1, N2, NDB, p, n, nmax, e))