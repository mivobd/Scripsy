% Initialization
clearvars
close all
clc
if ~isfolder('data')
    mkdir('data') % directory creation/check -> simulation results stored
else
    delete('data/*') % purge old results -> no between-run mismatch
end

% Parameters
    % Equipment dimensions
    A = 16.66; % dm^{-2}, reactor cross-sectional area
    V_C = 10; % L, volume of the cooling jacket's contents

    % Heat transfer
    rhoCp = 239; % J L^{-1} K^{-1}, product of the density and heat capacity of reactor contents
    rhoCp_C = 4175; % J L^{-1} K^{-1}, product of the density and heat capacity of coolant
    UA_C = 5e4; % J min^{-1} K^{-1}, product of the heat transfer coefficient and area available for heat transfer

    % Reaction thermodynamics
    k_0 = 7.2e10; % min^{-1}, Arrhenius constant
    deltaH = -5e4; % J mol^{-1}, heat of reaction
    EoR = 8750; % K, ratio of activation energy to the gas constant

    % Valve equations
    Cv = 44.4; % m^{3.5} kg^{-0.5}, reactor line valve coefficient
    Cv_C = 9.5; % m^{3.5} kg^{-0.5}, coolant line valve coefficient
    g_s = 1; % unitless, specific gravity of the fluid

% Variables
    % Disturbance variables
    SS.Q_F = 100; % L min^{-1}, flow rate of reactor feed stream (defined by nominal value of Q provided, since at SS Q=Q_F)
    SS.C_AF = 1.0; % mol L^{-1}, concentration of species A in the reactor feed stream
    SS.T_F = 320; % K, reactor feed stream temperature
    SS.T_CF = 300; % K, coolant feed stream temperature
    SS.Q = SS.Q_F; %Cv .* 1 .* sqrt( 1.73e12 .* ( P_U(t) - P_D(t) ) ./ g_s ); % L min^{-1}, reactor outlet stream flow rate
    SS.Q_C = 15; % L min^{-1}, coolant flow rate
    SS.P_D = 1.1e5; % Pa, reactor line downstream pressure
    SS.P_CU = 1.7e5; % Pa, coolant line upstream pressure
    SS.P_U = 2.41e5; % Pa, reactor line upstream pressure
    SS.P_CD = 1.01e5; % Pa, coolant line downstream pressure

    % Manipulated variables
    SS.lQ = SS.Q ./ ( Cv .* sqrt( 1.73e-12 .* ( SS.P_U - SS.P_D ) ./ g_s) .* 1000 );  
    SS.lQ_C = SS.Q_C ./ ( Cv .* sqrt( 1.73e-12 .* ( SS.P_CU - SS.P_CD ) ./ g_s) .* 1000 );

% ODEs
dXdt = @(t, X) [ ... where X = [C_A; % mol L^{-1}, concentration of species A inside reactor
                 ...            T; K, % temperature inside reactor 
                 ...            T_C; % K, temperature of coolant inside cooling jacket
                 ...            h % dm, level in reactor
                 ...            ]
    -k_0 .* exp( - EoR ./ X(2) ) .* X(1) + ( SS.Q_F .* SS.C_AF - SS.Q .* X(1) ) ./ ( A .* X(4) ); 
    k_0 .* exp( - EoR ./ X(2) ) .* X(1) .* ( - deltaH ) ./ rhoCp + ( SS.Q_F .* SS.T_F - SS.Q .* X(2) ) ./ ( A .* X(4) ) + UA_C .* ( X(3) - X(2) ) ./ ( rhoCp .* A .* X(4) );
    SS.Q_C .* ( SS.T_CF - X(3) ) ./ V_C + UA_C .* ( X(2) - X(3) ) ./ ( rhoCp_C .* V_C );
    ( SS.Q_F - SS.Q ) ./ A
];

X_nominal = [... % initial guesses, values from Wong, J. C. Y.. 2006. Fault detection applications of alternative mean-centering pattern matching in non-repeated multivariate time-series data trajectories.
    0.037; ...
    402.35; ...
    345.44; ...
    6.0 ...
];

% Steady state values
X_SS = fsolve(@(X) dXdt(0, X), X_nominal, optimset('Display', 'off')); % first estimate from fsolve is not sufficient

tspan = linspace(0, 100, 1000);
[~, X] = ode15s(dXdt, tspan, X_nominal); % using ode15s because of the system of equations displays stiff behaviour
SS.C_A = X(end, 1);
SS.T = X(end, 2);
SS.T_C = X(end, 3);
SS.h = X(end, 4);
 
% Controllers
    % Measurement transmitter gains
    Km_Q = 0.08; % mA min L^{-1}, 
    Km_Q_C = 8./15; % mA min L^{-1}, 
    Km_T = 8./402.35; % mA K^{-1}, 
    Km_h = 4./3; % mA dm^{-1}, 

    % Valve dynamics
    K = 1./16; % mA^{-1}, valve gain from controller signal
    tau = 2./60; % min, time constant for valve change from controller signal

    % h-Q cascade
    Kc_Q = 0.2; % mA min L^{-1}, gain of the inner loop h-Q cascade controller
    TI_Q = 1.7/60; % min, integral time of the inner loop h-Q cascade controller
    Kc_h = -3.0; % L min^{-1} dm^{-1}, gain of the outer loop h-Q cascade controller
    TI_h = 1.5; % min, integral time of the outer loop h-Q cascade controller

    % T-Q_C cascade
    Kc_Q_C = 0.2; % mA min L^{-1}, gain of the inner loop T-Q_C cascade controller
    TI_Q_C = 1.7/60; % min, integral time of the inner loop T-Q_C cascade controller
    Kc_T = -2.0; % L min^{-1} K^{-1}, gain of the outer loop T-Q_C cascade controller
    TI_T = 52.2/60; % min, integral time of the outer loop T-Q_C cascade controller

% Disturbances
D.details = readmatrix("fault_details.txt", 'OutputType', 'string');

%%
% Data generation
model.time = 120; % min, 
model.points = 120 ./ (5./60); % measurements recorded, will later be set to 1024

D.time = model.time - 5 ./ 60 .* ( 1024 - 1 ); % min, when the disturbance is effected
model.d_time = model.time; % min, will later become disturbance window size

model.runs = 6; % number of simulation runs

    % Simulation
    w = warning('off', 'MATLAB:structOnObject'); % unnecessary warning: revealing an object's hidden implementation details
    for ii = 1:model.runs
        rng("shuffle");
        D.fault = D.details(randi([1, numel(D.details(:, 1))]), 1)
        if ~ismember(D.fault, ["F3", "F13"])
            rng("shuffle");
            D.val = ( 0.25 + rand() ) .* double(D.details( D.details(:, 1) == D.fault, 2)); % disturbance magnitude varies between 25 % and 125 % of its nominal value
        end

        D.active = D.fault;
        output = sim("CSTR_Simulink.slx");
        D.active = "";

        results_struct = struct(output); % simulation output = object -> structure -> (names of variables measured) and (values) stored as dictionary
        model.results = dictionary( string(fieldnames(results_struct.Data)), struct2cell(results_struct.Data) );
        model.results("fault") = {cell(num2cell(double(char(D.fault))))};
        model.results("magnitude") = {D.val};
        model.results("fault time") = {D.time};
        writedictionary( model.results, "data/run_"+ii+".json");
    end
    warning(w); % removing the warning suppression to ensure it is not triggered where not expected

%%
plotvars = ["h", "Q", "T", "Q_C","C_A";
            "dm", "L min^{-1}", "K", "L min^{-1}", "mol L^{-1}"];
numPlots = numel(plotvars(1, :));

for jj = 1:model.runs
    data.results = readdictionary("data/run_"+jj+".json"); % results = keys: string, values: 1x1 cell array: 1024x1 cell
    data.fault = char(cell2mat(data.results{"fault"})');

    data.window = 200; % number of data points considered for the moving window average
    
    figure(jj)
    tile = tiledlayout(numPlots,1);
    tile.Title.String = data.fault + " | Magnitude = " + string(data.results("magnitude") + " " + D.details(D.details(:,1) == data.fault, 3));
    tile.Title.Interpreter = 'none';

    for ii = 1:numPlots
        nexttile
        data.t = cell2mat(data.results{"t"}); % curly brackets: indexes into cell array -> cell2mat: converts cell array to a matrix
        data.X = cell2mat(data.results{plotvars(1, ii)});

        plot(data.t, data.X, "k-") 
        hold on
        plot(data.results{"fault time"} .* [1, 1], [max(data.X), min(data.X)], "r-")
        plot(data.t, movmean(data.X, data.window), "b--")
        xlim([0, model.d_time])
        ylabel( plotvars(1, ii) + " (" + plotvars(2, ii) + ")" )
        xticks({})
        hold off
        
        data.mean = movmean(data.X, data.window);
    end
    xticks(linspace(0, model.d_time, (model.points-1)/8*5/60))
    xtickformat('%.2f')
    tile.Padding = 'compact';
    tile.TileSpacing = 'compact';

end
