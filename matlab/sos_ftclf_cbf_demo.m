%==========================================================
% run_sos_ftclf_cbf_demo.m
%
% Calls the SOS synthesis function, loads the synthesized
% controller, and simulates/plots trajectories from multiple
% initial conditions.
%
% Assumes you saved the synthesis code as:
%   sos_vanishing_clf_cbf_controller_ft.m
% (the function inside must be named
%   sos_vanishing_clf_cbf_controller_ft )
%
% Requires: YALMIP in path. For synthesis: MOSEK recommended.
%==========================================================

clear; clc; close all;

%-------------------------
% 1) Run SOS synthesis (creates sos_controller_ftclf.mat)
%-------------------------
sos_vanishing_clf_cbf_controller_ft();

%-------------------------
% 2) Load synthesized controller and parameters
%-------------------------
load('sos_controller_ftclf.mat', ...
    'u1_sol','u2_sol','RC','rB','T');

%-------------------------
% 3) Setup simulation utilities
%-------------------------
% IMPORTANT: create matching YALMIP symbols for replace()
sdpvar x1 x2 t

% Vanishing schedule alpha(t) (Algorithm-1 behavior inside B)
qalpha = 2;
alpha  = @(tt) 1/(1+tt)^qalpha;

% Controller evaluator u(t,x)
u_eval = @(tt,xx) [ ...
    double(replace(u1_sol, [x1 x2 t], [xx(1) xx(2) tt])); ...
    double(replace(u2_sol, [x1 x2 t], [xx(1) xx(2) tt])) ];

% Algorithm-1 vanishing inside B
apply_u = @(tt,xx) ( (xx(1)^2 + xx(2)^2) <= rB^2 ) .* (alpha(tt)*u_eval(tt,xx)) ...
                 + ( (xx(1)^2 + xx(2)^2) >  rB^2 ) .* (u_eval(tt,xx));

% Closed-loop ODE: xdot = -x + u(t,x)
odefun = @(tt,xx) (-xx + apply_u(tt,xx));

% Simulation horizon
tspan   = [0 40];
odeopts = odeset('RelTol',1e-7,'AbsTol',1e-9);

%-------------------------
% 4) Initial conditions to test (edit as desired)
%-------------------------
X0 = [ ...
     0  26;     % your paper example
   -10  30;
   -25  15;
    10  35;
   -40  50;
    -5  10;
   15   5;
   -5  -5
]';

%-------------------------
% 5) Simulate and plot
%-------------------------
figure; hold on; grid on;
xlabel('x_1'); ylabel('x_2');
title('SOS FT-CLF/CBF controller trajectories (with vanishing inside B)');

% Plot outer ball C and inner ball B
th = linspace(0,2*pi,500);
plot(RC*cos(th), RC*sin(th), '--', 'LineWidth', 1.0);
plot(rB*cos(th), rB*sin(th), ':',  'LineWidth', 1.0);

% Plot obstacle boundary hU=0 (same polynomial as synthesis)
[X1,X2] = meshgrid(linspace(-60,60,500), linspace(-60,60,500));
HU = 4*X1.^4 ...
    -20*X1.^2.*(X2-10) - 13*X1.^2 ...
    +25*(X2-10).^2 + 35*(X2-10) - 2;
contour(X1,X2,HU,[0 0],'LineWidth',1.2);

% Simulate each initial condition
traj_handles = gobjects(1, size(X0,2));
for k = 1:size(X0,2)
    x0 = X0(:,k);
    [tg,xg] = ode45(odefun, tspan, x0, odeopts);
    traj_handles(k) = plot(xg(:,1), xg(:,2), 'LineWidth', 1.4);
    plot(x0(1), x0(2), 'o', 'MarkerSize', 6, 'LineWidth', 1.2);
end

legend([traj_handles(1) ...
        plot(nan,nan,'--') plot(nan,nan,':')], ...
       'trajectories','outer ball C','inner ball B', ...
       'Location','best');

%-------------------------
% 6) Optional: time-series plots for one trajectory
%-------------------------
k_pick = 1;  % which IC to inspect
x0 = X0(:,k_pick);
[tg,xg] = ode45(odefun, tspan, x0, odeopts);

U = zeros(length(tg),2);
alpha_vals = zeros(length(tg),1);
insideB = false(length(tg),1);

for i = 1:length(tg)
    tt = tg(i);
    xx = xg(i,:)';
    u_nom = u_eval(tt,xx);
    insideB(i) = (xx(1)^2 + xx(2)^2) <= rB^2;
    alpha_vals(i) = insideB(i)*alpha(tt) + (~insideB(i))*1;
    U(i,:) = apply_u(tt,xx)';
end

figure; grid on; hold on;
plot(tg, xg(:,1), 'LineWidth', 1.4);
plot(tg, xg(:,2), 'LineWidth', 1.4);
xlabel('t'); ylabel('state');
title(sprintf('State vs time (IC #%d)', k_pick));
legend('x_1','x_2');

figure; grid on; hold on;
plot(tg, U(:,1), 'LineWidth', 1.4);
plot(tg, U(:,2), 'LineWidth', 1.4);
xlabel('t'); ylabel('u');
title(sprintf('Control vs time (IC #%d)', k_pick));
legend('u_1','u_2');

figure; grid on;
plot(tg, alpha_vals, 'LineWidth', 1.4);
xlabel('t'); ylabel('\alpha(t) multiplier');
title('Vanishing multiplier applied (1 outside B, \alpha(t) inside B)');
