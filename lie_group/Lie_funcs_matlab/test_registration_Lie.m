clc,
clear, 
close ALL

% v2skew = @(v) [0,-v(3),v(2); v(3),0,-v(1); -v(2),v(1),0];
% v2skew([1,2,3])

num_points = 6;
x_true = randn(3,num_points); % 3D points

rotvec_exact = pi * randn(3,1);
C_exact = expm(v2skew(rotvec_exact));
y_true = C_exact * x_true; % rotated (transformed) 3D points

% Add noise
sigma = 0.1; % standard deviation of noise
noise = sigma * randn(3,num_points);
y = y_true + noise;

% Before alignment / registration
figure,
plot3(x_true(1,:),x_true(2,:),x_true(3,:),'-ob')
hold on
plot3(y(1,:),y(2,:),y(3,:),'-or')
axis vis3d
daspect([1 1 1])
grid
xlabel('X'), ylabel('Y'), zlabel('Z')
title('Before optimization')


%% Optimization in R^3 parameter space: search for the rotation vector
% Yes, for this registration problem with rotations there is a closed-form
% solution. Ignore it.
disp('--------');
disp('Optimization in R^3 parameter space: search for the rotation vector');

% Objective function: some distance between point clouds
costf = @(rotv) cost_func(rotv,x_true,y);

% Optimize
options = optimoptions('fminunc','Display','iter', 'PlotFcn','optimplotfval');
r0 = [0,0,0]' % initial parameter vector in the optimization
%r0 = rotvec_exact + randn(3,1)
[r,fval,exitflag,output] = fminunc(costf,r0,options);
disp('best rotation vector:');
r

% Get rotation matrix from best rotation vector
C_param_space_search = expm(v2skew(r));
disp('Distance between exact and estimated rotation');
delta_param_R3 = norm(skew2v(logm(C_exact'*C_param_space_search)))


%% Optimization using Lie formulation. Newton's method adapted to rotations
disp('--------');
disp('Optimization using Lie-sensitive perturbations');
num_iters = 10;
cost = -ones(num_iters+1,1);
C_op = expm(v2skew(r0));
cost(1) = squared_distance(C_op*x_true,y);
disp(['iter: ' num2str(0), '  cost = ' num2str(cost(1))]);

% Iterate
for iter = 1:10

    % Compute the linear system Hessian * perturbation = - gradient 
    A = zeros(3,3);
    b = zeros(3,1);
    for j=1:num_points
        zj = C_op * x_true(:,j);
        A = A + v2skew(zj)'*v2skew(zj);
        b = b + v2skew(zj)'*(zj - y(:,j));
    end
    % Solve for the perturbation epsilon in the linear system of equations
    epsilon = A \ b;
    
    % Update "operating point", rotation matrix C_op
    C_op = expm(v2skew(epsilon)) * C_op;
    
    % Compute cost, to show how it evolves
    cost(1+iter) = squared_distance(C_op*x_true,y); % compute new cost
    disp(['iter: ' num2str(iter), '  cost = ' num2str(cost(1+iter)), '  norm(epsilon) = ' num2str(norm(epsilon))]);
    
    % Display evolution of the optimization process
    y_pred = C_op*x_true;
    figure(100),
    plot3(y(1,:),y(2,:),y(3,:),'-or')
    hold on
    plot3(y_pred(1,:),y_pred(2,:),y_pred(3,:),'-ob')
    hold off
    title(['iter: ' num2str(iter)])
    axis vis3d
    daspect([1 1 1])
    grid on
    xlabel('X'), ylabel('Y'), zlabel('Z')
    drawnow
    pause(0.5);

end
title('After optimization')

figure, plot(0:num_iters, cost)
grid on
xlabel('iteration')
title('Lie-sensitive perturbations: Evolution of cost value')

disp('Distance between exact and estimated rotation');
delta_param_Lie = norm(skew2v(logm(C_exact'*C_op)))

% The solutions obtained by the two methods are numerically very close:
disp('Distance between both estimated rotations');
delta_rots = norm(skew2v(logm(C_param_space_search'*C_op)))
