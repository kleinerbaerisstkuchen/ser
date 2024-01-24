function res = cost_func(rotv,x,y)
C = expm(v2skew(rotv)); % rotation matrix
res = squared_distance(y, C*x);
