%% Housekeeping (settings)
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
% simulink file open to view the scopes as it runs is still recommended (I
% especially recommend docking the 6 scopes for h, Q, lQ, T, Q_C, and lQ_C)

% Are you generating a large data set (comparable, or the same, in size as
% used by Johannesmeyer?
runs.full = 1; % boolean, generate the full dataset (Snapshot: one for each operating condition; Historical: 463 total, 54 NOC) or not
runs.NOC = 54; % integer, number of NOC windows
runs.fault = 409; % integer, number of windows with faults
runs.control = 0; % boolean, control (1) or no control (0) (may be changed to generate data for both automatically, depending on requirements down the line)

% Below (until next section) only applies if you are not generating a full
% data set (settings below are overridden/ignored for full dataset
% generation)
model.runs = 1; % integer, number of simulation runs to be conducted
model.folder = "data"; % string, where results are going to be saved 

% Controller settings
model.control_arrangement = 0; % boolean, true for 1/TI, false for TI (implementation inside the Simulink controller block)
model.control.T = 1; % boolean, true to close the temperature cascade control loop, false for no temperature cascade control
model.control.h = 1; % boolean, true to close the level cascade control loop, false for no level cascade control
model.contoller_parameters = 1; % boolean, true to use the retuned values, false to use the values from Wong

% Noise settings
model.noise.M = 1; % boolean, measurement white noise active or not 
model.noise.AR = 1; % boolean, auto-regressive feed noise active or not

% Disturbance settings
model.disturb = 1; % boolean, incur disturbances or not
D.force.use = 1; % boolean, force a specific disturance (remember to switch on model.disturb)
D.force.fault = "F6"; % string, fault ID, the fault you want to incur (refer to fault_details.txt)
D.force.mag = 5; % double, units as per fault ID (fault_details.txt contains all relevant info)

%% Initialization
if runs.full == 1
    % ensuring noise is present
    model.noise.M = 1;
    model.noise.AR = 1;

    % controller settings
    model.control_arrangement = 0;
    model.contoller_parameters = 1;
    if runs.control == 0
        model.control.T = 0;
        model.control.h = 0;
    else
        model.control.T = 1;
        model.control.h = 1;
    end
    
    % directory creation/check -> simulation results stored here
    if ~isfolder("H"+runs.control)
        mkdir("H"+runs.control)
    end
    if ~isfolder("S"+runs.control)
        mkdir("S"+runs.control)
    end
else
    % directory creation/check -> simulation results stored here
    if ~isfolder(model.folder)
        mkdir(model.folder)
    end
end

% Model details
model.dt = 5./60; % measurement interval
model.time = 120; % min, how long the simulation will run for
model.points = model.time ./ model.dt; % number of measurements recorded

D.time = model.time - model.dt .* ( 1024 - 1 ); % min, when the disturbance is effected
model.d_time = model.time - D.time; % min, will later become disturbance window size

model.tauf = model.dt.*2; % analogue filter time constant, should be roughly half of the measurement interval (Seborg et al.)
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
    g_s = 1; % ~, specific gravity of the fluid
    N = 0.0865 * 1000 / 60 / sqrt(1000); % L min^{-1} Pa^{-0.5}, units conversion factor

    % Valve dynamics
    K = 1./16; % mA^{-1}, valve gain from controller signal
    tau = 2./60; % min, time constant for valve change from controller signal

    if model.contoller_parameters == 1
        % h-Q cascade tuning parameters
        Kc_Q = 0.2/2.5; % mA min L^{-1}, gain of the inner loop h-Q cascade controller
        TI_Q = 1.7/60*15; % min, integral time of the inner loop h-Q cascade controller
        Kc_h = -3.0*15; % L min^{-1} dm^{-1}, gain of the outer loop h-Q cascade controller
        TI_h = 1.5/60*15; % min, integral time of the outer loop h-Q cascade controller
    
        % T-Q_C cascade tuning parameters
        Kc_Q_C = 0.2; % mA min L^{-1}, gain of the inner loop T-Q_C cascade controller
        TI_Q_C = 1.7/60*35; % min, integral time of the inner loop T-Q_C cascade controller
        Kc_T = -2.0/2; % L min^{-1} K^{-1}, gain of the outer loop T-Q_C cascade controller
        TI_T = 52.2/60; % min, integral time of the outer loop T-Q_C cascade controller
    else
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
    end

    if model.control_arrangement == 1 % changing control configuration for testing (hopefully this is here temporarily... if this is still here in production then know I suffered)
        TI_Q = 1./TI_Q;
        TI_h = 1./TI_h;
        TI_Q_C = 1./TI_Q_C;
        TI_T = 1./TI_T;
    end

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
    SS.P_D = 1.1e5; % Pa, reactor line downstream pressure
    SS.P_U = 2.41e5; % Pa, reactor line upstream pressure

    SS.Q_C = 15; % L min^{-1}, coolant flow rate
    SS.T_CF = 300; % K, coolant feed stream temperature
    SS.P_CU = 1.7e5; % Pa, coolant line upstream pressure
    SS.P_CD = 1.01e5; % Pa, coolant line downstream pressure

    
    % Valve sizing
    SS.Q = SS.Q_F; % L min^{-1}, reactor outlet stream flow rate
    SS.lQ = 0.5; % ~, valve level for the reactor outlet flow control valve at steady state
    SS.lQ_C = 0.5; % ~, valve level for the cooling jacket coolant inlet flow control valve at steady state

    Cv_Q = SS.Q ./ ( SS.lQ .* N .* sqrt( ( SS.P_U - SS.P_D ) ./ g_s) ); % ~, the valve coefficient for the reactor outlet flow valve
    Cv_Q_C = SS.Q_C ./ ( SS.lQ_C .* N .* sqrt( ( SS.P_CU - SS.P_CD ) ./ g_s ) ); % ~, the valve coefficient for the reactor cooling jacket coolant inlet valve

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

% % Can check to see that the new SS is reached
% figure(1)
% plot(tspan, X(:, 1))
% figure(2)
% plot(tspan, X(:, 2))
% figure(3)
% plot(tspan, X(:, 3))
% figure(4)
% plot(tspan, X(:, 4))

X_nominal = [... % initial guesses, values from Wong, J. C. Y.. 2006. Fault detection applications of alternative mean-centering pattern matching in non-repeated multivariate time-series data trajectories.
    SS.C_A; ...
    SS.T; ...
    SS.T_C; ...
    SS.h ...
    ];

% Steady state values
tspan = linspace(0, 100, 10000);
[~, X] = ode23s(dXdt, tspan, X_nominal); % ODE system is stiff. Tested them all and 23s gave the best performance.
SS.C_A = X(end, 1);
SS.T = X(end, 2);
SS.T_C = X(end, 3);
SS.h = X(end, 4);

% % Can check to see that the new SS is reached
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
    runs.snapshot = numel(D.details(:,1)) + 1;
    model.runs = runs.NOC + runs.fault + runs.snapshot;

    % Faults are considered
    D.fault = strings(model.runs, 1);
    D.mags = ones(model.runs, 1);

    D.fault(:) = "N";
    D.fault(2:runs.snapshot) = D.details(:,1);

    for ii = 2:runs.snapshot
        if ~ismember(D.fault(ii), ["F3", "F13"]) % these don't have a magnitude, so are treated differently
            rng("shuffle");
            D.mags(ii) = 0.75 .* double(D.details( D.details(:, 1) == D.fault(ii), 2));
        end
    end

    for ii = runs.snapshot+runs.NOC+1:model.runs
        % List of faults
        rng("shuffle");
        D.fault(ii) = D.details(randi([1, numel(D.details(:, 1))]), 1);

        % List of magnitudes (same order as faults)
        if ~ismember(D.fault(ii), ["F3", "F13"]) % these don't have a magnitude, so are treated differently
            rng("shuffle");
            D.mags(ii) = ( 0.25 + rand() ) .* double(D.details( D.details(:, 1) == D.fault(ii), 2)); % disturbance magnitude varies between 25 % and 125 % of its nominal value
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
    if model.disturb == true || runs.full == true
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

    if runs.full == 1 && ii <= runs.snapshot
        if isfile("S"+runs.control+"/"+D.active+".json")
            delete("S"+runs.control+"/"+D.active+".json")
        end
        writedictionary(model.results, "S"+runs.control+"/"+D.active+".json");
    else
        if isfile("H"+runs.control+"/run_"+(ii-runs.snapshot)+".json")
            delete("H"+runs.control+"/run_"+(ii-runs.snapshot)+".json")
        end
        writedictionary(model.results, "H"+runs.control+"/run_"+(ii-runs.snapshot)+".json");
    end
end
warning(w); % removing the warning suppression since we no longer expect this warning to be thrown (therefore need to know about it, if it does happen)

%%
clc
