%% Parameters
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

%%
