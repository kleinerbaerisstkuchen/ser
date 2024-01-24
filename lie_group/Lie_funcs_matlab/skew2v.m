function v = skew2v(M)
v = 0.5 * [M(3,2)-M(2,3); M(1,3)-M(3,1); M(2,1)-M(1,2)];

