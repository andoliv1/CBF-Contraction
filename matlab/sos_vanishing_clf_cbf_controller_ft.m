 function sos_vanishing_clf_cbf_controller_ft()
%======================================================================
% sos_vanishing_clf_cbf_controller_ft.m
%
% Offline SOS synthesis of a polynomial controller u(x,t) enforcing:
%   (1) CBF for outer ball:     hC(x) >= 0
%   (2) CBF for obstacle-safe:  hU(x) >= 0   (YOUR POLYNOMIAL)
%   (3) Finite-time set-Lyapunov condition (classical):
%         d/dt V(t,x(t)) <= -cV * V(t,x(t))^gamma,  gamma in (0,1),
%       whenever x(t) notin S_t.
%
% SOS implementation:
%   - Let S_t = { hS(x,t) >= 0 } be the tube.
%   - Define violation s(x,t) := -hS(x,t). Then x notin S_t <=> s >= 0.
%   - Use V(t,x) = (max{s,0})^qV (C^1 for qV>=2), but we NEVER encode max.
%   - On the region s>=0, V = s^qV and V^gamma = s^pV if gamma=pV/qV.
%   - Enforce polynomial inequality on s>=0:
%         qV*s^(qV-1)*sdot + cV*s^pV <= 0
%     where sdot = ds/dt + Lf s + Lg s u.
%
% Tube schedule:
%   To keep SOS polynomial in (x,t), we use r(t)=r0/(1+t)^pr and clear
%   denominators. (This is NOT about polynomial decay of V.)
%
% Requires: YALMIP + SDP solver (MOSEK recommended).
%======================================================================

clear; clc;
yalmip('clear');

%--------------------------
% Parameters (edit as needed)
%--------------------------
RC   = 60;      % outer ball radius for C
rB   = 14;      % inner ball radius B (used to restrict constraints outside B)
T    = 25;      % SOS time horizon: t in [0,T]

deg_u = 3;      % degree of controller polynomial u(x,t)
deg_s = 4;      % degree of SOS multipliers

kC = 1.0;       % class-K gain for hC CBF: kappa(h)=k*h
kU = 1.0;       % class-K gain for hU CBF

% Finite-time parameters (choose integers pV,qV with 0<pV<qV)
qV = 2;         % qV>=2
pV = 1;         % 0<pV<qV, so gamma=pV/qV in (0,1)
cV = 2.0;
gamma = pV/qV;
fprintf('Finite-time choice: gamma = %g (pV=%d, qV=%d)\n', gamma, pV, qV);

% Tube parameters: center (r(t), -r(t)), radius^2=10
RS2 = 10;
r0  = 50;
pr  = 2;        % r(t)=r0/(1+t)^pr

% Optional: certify inner ball is obstacle-free (hU >= eps on ||x||<=rB)
enforce_B_safe = true;
eps_safe = 1e-3;

% Optional: run simulation after synthesis
do_sim = true;

%--------------------------
% Variables
%--------------------------
sdpvar x1 x2 t
vars = [x1 x2 t];

%--------------------------
% Dynamics: xdot = f(x) + u  (example: f=-x, g=I)
%--------------------------
f = [-x1; -x2];
g = eye(2);

%--------------------------
% YOUR obstacle polynomial hU(x1,x2) (SAFE set is hU >= 0)
%--------------------------
hU = 4*x1^4 ...
    -20*x1^2*(x2-10) - 13*x1^2 ...
    +25*(x2-10)^2 + 35*(x2-10) - 2;

%--------------------------
% Outer ball hC (C = {hC >= 0})
%--------------------------
hC = RC^2 - x1^2 - x2^2;

% Outside inner ball indicator: enforce constraints only when outside B
hBout = (x1^2 + x2^2) - rB^2;   % >=0 means outside B

% Time interval polynomial: >=0 on [0,T]
hT = t*(T - t);

%--------------------------
% Tube S_t with algebraic schedule and cleared denominators
% r(t)=r0/(1+t)^pr, define w=(1+t)^pr, then:
%   (x1-r(t))^2 + (x2+r(t))^2 <= RS2
% <=> (w*x1-r0)^2 + (w*x2+r0)^2 <= w^2*RS2
% Let hS >=0 inside tube.
%--------------------------
w  = (1+t)^pr;
hS = (w^2)*RS2 - (w*x1 - r0)^2 - (w*x2 + r0)^2; % >=0 inside tube
s  = -hS;                                        % >=0 outside tube

%--------------------------
% Controller polynomial u(x,t)
% IMPORTANT FIX:
%   In YALMIP, [p,c]=polynomial(vars,deg) returns polynomial p directly.
%--------------------------
[u1, c1] = polynomial(vars, deg_u);
[u2, c2] = polynomial(vars, deg_u);
u = [u1; u2];

%--------------------------
% CBF Lie derivatives
%--------------------------
dhCdx = jacobian(hC, [x1 x2]);       % 1x2
Lf_hC = dhCdx*f;                    % scalar
Lg_hC = dhCdx*g;                    % 1x2

dhUdx = jacobian(hU, [x1 x2]);       % 1x2
Lf_hU = dhUdx*f;
Lg_hU = dhUdx*g;

% CBF constraints: Lf h + Lg h u + k*h >= 0
qC = Lf_hC + Lg_hC*u + kC*hC;        % scalar
qU = Lf_hU + Lg_hU*u + kU*hU;        % scalar

%--------------------------
% Finite-time set-Lyapunov constraint on s>=0 (outside tube)
% sdot = ds/dt + Lf s + Lg s u
% enforce: qV*s^(qV-1)*sdot + cV*s^pV <= 0  on s>=0
% <=> pFT := -( qV*s^(qV-1)*sdot + cV*s^pV ) >= 0 on domain + s>=0
%--------------------------
dsdt = jacobian(s, t);
dsdx = jacobian(s, [x1 x2]);         % 1x2
Lf_s = dsdx*f;
Lg_s = dsdx*g;

sdot = dsdt + Lf_s + Lg_s*u;         % scalar
pFT  = -( qV*(s^(qV-1))*sdot + cV*(s^pV) );  % scalar

%--------------------------
% SOS constraints with multipliers (Putinar)
% Domain K for CBFs:
%   hC>=0, hU>=0, hBout>=0, hT>=0
% FT constraint domain:
%   same + s>=0
%--------------------------
F = [];

% ---- qC >= 0 on K
[sC1,~] = polynomial(vars, deg_s);
[sC2,~] = polynomial(vars, deg_s);
[sC3,~] = polynomial(vars, deg_s);
[sC4,~] = polynomial(vars, deg_s);
F = [F, sos(sC1), sos(sC2), sos(sC3), sos(sC4)];
F = [F, sos( qC - sC1*hC - sC2*hU - sC3*hBout - sC4*hT )];

% ---- qU >= 0 on K
[sU1,~] = polynomial(vars, deg_s);
[sU2,~] = polynomial(vars, deg_s);
[sU3,~] = polynomial(vars, deg_s);
[sU4,~] = polynomial(vars, deg_s);
F = [F, sos(sU1), sos(sU2), sos(sU3), sos(sU4)];
F = [F, sos( qU - sU1*hC - sU2*hU - sU3*hBout - sU4*hT )];

% ---- pFT >= 0 on K plus s>=0
[sV1,~] = polynomial(vars, deg_s);
[sV2,~] = polynomial(vars, deg_s);
[sV3,~] = polynomial(vars, deg_s);
[sV4,~] = polynomial(vars, deg_s);
[sV5,~] = polynomial(vars, deg_s);   % multiplier for s>=0
F = [F, sos(sV1), sos(sV2), sos(sV3), sos(sV4), sos(sV5)];
F = [F, sos( pFT - sV1*hC - sV2*hU - sV3*hBout - sV4*hT - sV5*s )];

% ---- Optional: certify inner ball B is obstacle-safe
if enforce_B_safe
    % hBin >= 0 represents ||x||<=rB
    hBin = rB^2 - (x1^2 + x2^2);
    [sB,~] = polynomial([x1 x2], deg_s);
    F = [F, sos(sB)];
    F = [F, sos( hU - eps_safe - sB*hBin )];
end

%--------------------------
% Objective: small controller (proxy for min-norm around u_nom=0)
% Use sum of squares of controller coefficient vectors.
%--------------------------
obj = c1'*c1 + c2'*c2;

%--------------------------
% Solve
%--------------------------
opts = sdpsettings('solver','mosek','verbose',1);
sol  = solvesos(F,[],opts,[sC]);

disp('--------------------------------------------------');
disp(sol.info);
disp('--------------------------------------------------');

if sol.problem ~= 0
    warning('SOS solve not successful. Try increasing deg_u/deg_s or shrinking RC/T, or relaxing rB usage.');
end

% Substitute solved coefficients into u1,u2 (now functions of x,t only)
u1_sol = clean(value(u1), 1e-9);
u2_sol = clean(value(u2), 1e-9);

disp('u1(x,t) ='); disp(sdisplay(u1_sol));
disp('u2(x,t) ='); disp(sdisplay(u2_sol));

save('sos_controller_ftclf.mat', ...
    'u1_sol','u2_sol','RC','rB','T','RS2','r0','pr','kC','kU','cV','pV','qV','gamma');

%--------------------------
% Optional simulation with Algorithm-1 style vanishing inside B
%--------------------------
if do_sim
    % vanishing schedule alpha(t)
    qalpha = 2;
    alpha  = @(tt) 1/(1+tt)^qalpha;

    % controller evaluator (IMPORTANT: uses SAME sdpvars x1,x2,t)
    u_eval = @(tt,xx) [ ...
        double(replace(u1_sol, [x1 x2 t], [xx(1) xx(2) tt])); ...
        double(replace(u2_sol, [x1 x2 t], [xx(1) xx(2) tt])) ];

    % closed-loop ODE
    odefun = @(tt,xx) (-xx + apply_vanishing(tt,xx,u_eval,rB,alpha));

    % initial condition (example)
    x0 = [0; 26];
    tspan = [0 40];
    odeopts = odeset('RelTol',1e-7,'AbsTol',1e-9);

    [tg,xg] = ode45(odefun, tspan, x0, odeopts);

    figure; plot(xg(:,1), xg(:,2), 'LineWidth', 1.5); grid on;
    xlabel('x_1'); ylabel('x_2');
    title('Trajectory (SOS FT-CLF/CBF controller + vanishing in B)');
    hold on;

    % plot outer ball C and inner ball B
    th = linspace(0,2*pi,400);
    plot(RC*cos(th), RC*sin(th), '--', 'LineWidth', 1.0);
    plot(rB*cos(th), rB*sin(th), ':',  'LineWidth', 1.0);

    % plot obstacle boundary hU=0
    [X1,X2] = meshgrid(linspace(-60,60,350), linspace(-60,60,350));
    HU = 4*X1.^4 ...
        -20*X1.^2.*(X2-10) - 13*X1.^2 ...
        +25*(X2-10).^2 + 35*(X2-10) - 2;
    contour(X1,X2,HU,[0 0],'LineWidth',1.2);

    legend('trajectory','outer ball C','inner ball B','h_U=0');
end

end

%======================================================================
% Helper: apply Algorithm-1 vanishing inside B
%======================================================================
function u = apply_vanishing(t,x,u_eval,rB,alpha)
u = u_eval(t,x);
if (x(1)^2 + x(2)^2) <= rB^2
    u = alpha(t)*u;
end
end
