clc,
clear, 
close ALL

v2skew = @(v) [0,-v(3),v(2); v(3),0,-v(1); -v(2),v(1),0];
% v2skew([1,2,3])

x = [1,0,0]';
%C_mean = expm(pi*v2skew(randn(3,1)));
C_mean = eye(3);

sigma = .2; % standard deviation of noise
num_samples = 500;
e_samples = randn(3,num_samples);
%e_samples = sigma * e_samples;
e_samples = diag([sigma,sigma,2*sigma])* e_samples;

y = zeros(3, num_samples);
for k = 1:num_samples
    y(:,k) = expm(v2skew(e_samples(:,k))) * C_mean * x;
end

ym = C_mean * x;

figure,
sphere
axis equal
hold on, 
plot3(ym(1),ym(2),ym(3),'*k')
plot3(y(1,:),y(2,:),y(3,:),'.r')
axis vis3d
daspect([1 1 1])
grid on
xlabel('X'), ylabel('Y'), zlabel('Z')