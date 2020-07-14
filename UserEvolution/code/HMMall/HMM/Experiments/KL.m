function [dKL] = KL(mu0,mu1,A)
% Kullback Leibler divergence (KL) KL(p,q)
% Input: 
%     p: p.mu, p.sigma
%     q: q.mu, q.sigma
% Output: 
%     delta: the KL divergence between p and q.

mu0 = mu0/norm(mu0,1);
mu1 = mu1/norm(mu1,1);

k = length(mu0);
tmp = inv(A)*A;
dKL = trace(tmp) - k - log(det(tmp)) + 0.5*((mu1-mu0)'*inv(A)*(mu1-mu0)) + (0.5*((mu0-mu1)'*inv(A)*(mu0-mu1)));
%dKL = 0.5*((mu1-mu0)'*(mu1-mu0)-k) + (0.5*((mu0-mu1)'*(mu0-mu1)-k));
%dKL = dKL/(2*log(2));
dKL = dKL/2;