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

model.folder = "S"; % Where results are going to be saved 

% Are you generating a large data set (comparable, or the same, in size as
% used by Johannesmeyer?
runs.full = 0; % Generate the full dataset (463 total, 54 NOC) or not
runs.NOC = 54; % Number of NOC windows
runs.fault = 409; % Number of windows with faults

% Below only applies if you are not generating a full data set
model.runs = 1; % number of simulation runs to be conducted
model.disturb = 1; % boolean, incur disturbances or not
model.plot = 0; % boolean, plot after each run or not

model.noise.P = 0; % boolean, process white noise active or not
model.noise.M = 1; % boolean, measurement white noise active or not 
model.noise.AR = 1; % boolean, auto-regressive feed noise active or not

D.force.use = 1; % boolean, force a specific disturance (remember to switch on model.disturb)
D.force.fault = "F10"; % fault ID, the fault you want to incur (refer to fault_details.txt)
D.force.mag = 17236.9; % units as per fault ID (refer to fault_details.txt)

%% Initialization
if ~isfolder(model.folder)
    mkdir(model.folder) % directory creation/check -> simulation results stored here
end

% Model details
model.dt = 5./60; % measurement interval
model.time = 120; % min, how long the simulation will run for
model.points = model.time ./ model.dt; % number of measurements recorded

D.time = model.time - model.dt .* ( 1024 - 1 ); % min, when the disturbance is effected
model.d_time = model.time - D.time; % min, will later become disturbance window size

model.tauf = model.dt.*1; % analogue filter time constant, should be roughly half of the measurement interval (Seborg et al.)
model.autoreg = 0.8; % autoregression gain

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
    g_s = 1; % ~, specific gravity of the fluid

    % Valve dynamics
    K = 1./16; % mA^{-1}, valve gain from controller signal
    tau = 2./60; % min, time constant for valve change from controller signal

    % h-Q cascade tuning parameters
    Kc_Q = 0.2; % mA min L^{-1}, gain of the inner loop h-Q cascade controller
    TI_Q = 1.7/60; % min, integral time of the inner loop h-Q cascade controller
    Kc_h = -3.0; % L min^{-1} dm^{-1}, gain of the outer loop h-Q cascade controller
    TI_h = 1.5; % min, integral time of the outer loop h-Q cascade controller

    % T-Q_C cascade tuning parameters
    Kc_Q_C = 0.2; % mA min L^{-1}, gain of the inner loop T-Q_C cascade controller
    TI_Q_C = 1.7/60; % min, integral time of the inner loop T-Q_C cascade controller
    Kc_T = -2.0; % L min^{-1} K^{-1}, gain of the outer loop T-Q_C cascade controller
    TI_T = 52.2/60; % min, integral time of the outer loop T-Q_C cascade controller

    % Measurement transmitter gains
    Km_Q = 0.08; % mA min L^{-1}, for the outlet flow rate
    Km_Q_C = 8/15; % mA min L^{-1}, for the coolant flow rate
    Km_T = 8./402.35; % mA K^{-1}, for the reactor temperature
    Km_h = 4/3; % mA dm^{-1}, for the reactor level
    % unfortunately no other measurement gains were listed in Wong's thesis
    % (the primary source of Johannesmeyer's values at the time of writing
    % this - [26/04/2026])

% Disturbances
% Stored the specifics of each disturbance ID in a separate file to make
% changes more easily.
D.details = readmatrix("fault_details.txt", 'OutputType', 'string');

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
    level_change = 1; % valve is moved 100 % open from completely closed
    signal_to_valve_function = level_change / K; % mA

    SS.lQ = SS.Q / ( signal_to_valve_function / Kc_Q ); % ~, steady state valve level for the feed flow rate
    SS.lQ_C = SS.Q_C / ( signal_to_valve_function / Kc_Q_C ); % ~, steady state valve level for the coolant flow rate

    N.Q = ( SS.Q ./ 1000 ) ./ ( Cv .* SS.lQ .* sqrt( ( SS.P_U - SS.P_D ) ./ g_s) ); % conversion factor, sqrt(1.73e12) in Wong's thesis, assuming a typo then sqrt(1.73e-12)
    N.Q_C = ( SS.Q_C ./ 1000 ) ./ ( Cv .* SS.lQ_C .* sqrt( ( SS.P_CU - SS.P_CD ) ./ g_s ) ); % conversion factor, sqrt(1.73e12) in Wong's thesis, assuming a typo then sqrt(1.73e-12)
    
    % N.Q = sqrt(1.73e-12);
    % N.Q_C = sqrt(1.73e-12);
    % 
    % SS.lQ = ( SS.Q ./ 1000 ) ./ ( Cv .* N.Q .* sqrt( ( SS.P_U - SS.P_D ) ./ g_s ) );
    % SS.lQ_C = ( SS.Q_C ./ 1000 ) ./ ( Cv .* N.Q_C .* sqrt( ( SS.P_CU - SS.P_CD ) ./ g_s ) );
    % 
    % SS.lQ
    % SS.lQ_C 

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
tspan = linspace(0, 100, 10000);
[~, X] = ode23s(dXdt, tspan, X_nominal); % ODE system is stiff. Tested them all and 23s gave the best performance.
SS.C_A = X(end, 1);
SS.T = X(end, 2);
SS.T_C = X(end, 3);
SS.h = X(end, 4);

% Can check to see that the new SS is reached
% figure(1)
% plot(tspan, X(:, 1))
% figure(2)
% plot(tspan, X(:, 2))
% figure(3)
% plot(tspan, X(:, 3))
% figure(4)
% plot(tspan, X(:, 4))

%% Data Generation
% Fault definitions
if runs.full == true
    model.runs = runs.NOC + runs.fault;

    % Faults are considered
    D.fault = strings(model.runs, 1);
    D.mags = ones(model.runs, 1);

    D.fault(:) = "N";

    for ii = runs.NOC+1:model.runs
        % List of faults
        rng("shuffle");
        D.fault(ii) = D.details(randi([1, numel(D.details(:, 1))]), 1);

        % List of magnitudes (same order as faults)
        if ~ismember(D.fault(ii), ["F3", "F13"]) % these don't have a magnitude, so are treated differently (not ableist I swear...)
            rng("shuffle");
            D.mags(ii) = ( 0.25 + rand() ) .* double(D.details( D.details(:, 1) == D.fault(ii), 2)).*~D.force.use + D.force.mag.*D.force.use; % disturbance magnitude varies between 25 % and 125 % of its nominal value
        end
    end
elseif model.disturb == true
    % Faults are considered
    D.fault = strings(model.runs, 1);
    D.mags = ones(model.runs, 1);

    for ii = 1:model.runs
        % List of faults
        if D.force.use == true
            D.fault(ii) = D.force.fault;
        else
            rng("shuffle");
            D.fault(ii) = D.details(randi([1, numel(D.details(:, 1))]), 1);
        end
        
        % List of magnitudes (same order as faults)
        if ~ismember(D.fault(ii), ["F3", "F13"]) % these don't have a magnitude, so are treated differently (not ableist I swear...)
            rng("shuffle");
            D.mags(ii) = ( 0.25 + rand() ) .* double(D.details( D.details(:, 1) == D.fault(ii), 2)).*~D.force.use + D.force.mag.*D.force.use; % disturbance magnitude varies between 25 % and 125 % of its nominal value
        end
    end
else
    % No faults
    D.active = "N";
    D.val = 1;
end

% Simulation
w = warning('off', 'MATLAB:structOnObject'); % unnecessary warning: revealing an object's hidden implementation details
for ii = 1:model.runs

    % Setting the fault for the run
    if model.disturb == true
        D.active = D.fault(ii); 
        D.val = D.mags(ii);
    end

    output = sim("CSTR_Simulink.slx"); % execute simulation and save results object

    % Reformatting the results object into a structure, which can then be
    % saved into a .json file (alongside details about the fault)
    results_struct = struct(output);
    model.results = dictionary( string(fieldnames(results_struct.Data)), struct2cell(results_struct.Data) ); % nasty datatype manipulation
    model.results("measurements") = {keys(model.results)}; % recording the measurement set taken (makes life easier later, less manual entry)
    model.results("fault") = {cell(num2cell(double(char(D.active))))}; % forced to used cells for dictionary to be happy
    model.results("magnitude") = {D.val}; % the curly brackets ensure each dictionary entry is a 1x1 cell 
    model.results("fault time") = {D.time};

    if isfile(model.folder+"/run_"+ii+".json")
        delete(model.folder+"/run_"+ii+".json")
    end
    writedictionary(model.results, model.folder+"/run_"+ii+".json");
end
warning(w); % removing the warning suppression since we no longer expect this warning to be thrown (therefore need to know about it, if it does happen)

%% Deprecated
% Plots
if model.plot == true && runs.full == false % checking if we want to see plots, or if too many would need to be made
    plotvars = ["h", "Q", "T", "Q_C","C_A"; ... % the variables being plotted
                "dm", "L / min", "K", "L / min", "mol / L"]; % and their units
    numPlots = numel(plotvars(1, :));
    
    for jj = 1:model.runs
        data.results = readdictionary(model.folder+"/run_"+jj+".json"); % results = (keys: string), (values: 1x1 cell array: 1024x1 cell)
        data.fault = char(cell2mat(data.results{"fault"})'); % moulding the cell array back into a well-behaved string telling us the fault ID
    
        data.window = 250; % number of data points considered for the moving window average
        
        figure(jj)
        tile = tiledlayout(numPlots,1);
        % Plot title: Fault ID + magnitude + units
        if ~ismember(data.fault, ["F3", "F13"]) % again, we discrim- I mean treat these differently since they don't have a disturbance magnitude
            tile.Title.String = data.fault + " | Magnitude = " + string(data.results("magnitude") + " " + D.details(D.details(:,1) == data.fault, 3));
            tile.Title.Interpreter = 'none';
        else
            tile.Title.String = data.fault;
            tile.Title.Interpreter = 'none';
        end
    
        for ii = 1:numPlots
            nexttile
            data.t = cell2mat(data.results{"t"}); % curly brackets: indexes into (1x1) cell array (which was necessary for a happy dictionary definition), then we use cell2mat: converts cell array to a matrix
            data.X = cell2mat(data.results{plotvars(1, ii)}); % similar logic as above, but for each variable we want to plot
    
            % Plotting the data... finally
            plot(data.t, data.X, 'linestyle', '-', 'color', '#AAAAAA') 
            hold on
    
            % Indicating where the disturbance happened
            if max(data.X) == min(data.X)
                plot(data.results{"fault time"} .* [1, 1], [data.X .* 0.75, data.X .* 1.25], 'linestyle', '-', 'color', '#FF0000')
            else
                plot(data.results{"fault time"} .* [1, 1], [max(data.X), min(data.X)], 'linestyle', '-', 'color', '#FF0000')
            end
    
            % Showing what the moving window mean looks like
            plot(data.t, movmean(data.X, data.window), 'linestyle', '--', 'color', '#000FFF')
    
            % Making the formatting nice... hopefully
            xlim([0, model.d_time])
            ylabel( plotvars(1, ii) + " (" + plotvars(2, ii) + ")" )
            xticks({})
            hold off
    
        end
        % Only the bottom plot needs an x axis with a label and ticks
        % (there's a joke in there somewhere...)
        xticks(linspace(0, model.d_time, (model.points-1)/9*5/60))
        xtickformat('%.1f')
        tile.Padding = 'compact';
        tile.TileSpacing = 'compact';
    
    end
end
%%
clc

