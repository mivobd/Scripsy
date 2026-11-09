%% Housekeeping (settings)
clearvars
close all
clc

% generate PPCA model to validate BPCA EM algorithm
opt = statset('ppca');
opt.MaxIter = 1e4;

jsonStr = fileread("simulation_data/test_x.json");
x = jsondecode(jsonStr);
K = size(x);

[coeff, score, pcvar, mu, v, S] = ppca(x',K(1), 'Options', opt);

results.W = reshape(S.W, [], numel(S.W));
results.v = S.v;
[N, k] = size(S.W);
results.k = k;
1/results.v

if isfile("BPCA_models/MATLAB_PPCA.json")
    delete("BPCA_models/MATLAB_PPCA.json")
end
writestruct(results, "BPCA_models/MATLAB_PPCA.json");