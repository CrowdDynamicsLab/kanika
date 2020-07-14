function [lik,path,nelem] = viterbi_path_testdata(prior, transmat, obslik,index)
% VITERBI Find the most-probable (Viterbi) path through the HMM state trellis.
% path = viterbi(prior, transmat, obslik)
%
% Inputs:
% prior(i) = Pr(Q(1) = i)
% transmat(i,j) = Pr(Q(t+1)=j | Q(t)=i)
% obslik(i,t) = Pr(y(t) | Q(t)=i)
%
% Outputs:
% path(t) = q(t), where q1 ... qT is the argmax of the above expression.


% delta(j,t) = prob. of the best sequence of length t-1 and then going to state j, and O(1:t)
% psi(j,t) = the best predecessor state, given that we ended up in state j at t

scaled = 1;

T = size(obslik, 2);
prior = prior(:);
Q = length(prior);

delta = zeros(Q,T);
psi = zeros(Q,T);
path = zeros(1,T);
scale = ones(1,T);
lik = 0.0;
maxi = -1;
t=1;
delta(:,t) = prior .* obslik(:,t);
if scaled
  [delta(:,t), n] = normalise(delta(:,t));
  scale(t) = 1/n;
end
psi(:,t) = 0; % arbitrary value, since there is no predecessor to t=1
for t=2:T
  for j=1:Q
    [delta(j,t), psi(j,t)] = max(delta(:,t-1) .* transmat(:,j));
    delta(j,t) = delta(j,t) * obslik(j,t);
  end
  if scaled
    %'In scaled',max(delta(:,t))
    [delta(:,t), n] = normalise(delta(:,t));
    scale(t) = 1/n;
  end 
  if (t == index)
	[maxv,maxi] = max(delta(:,t));
  end
  if(t>index)
      	%(max(delta(:,t)))
	%'Scale',scale(t)
	%lik=lik + log(scale(t));
	%lik = lik + delta(maxi,t);
	[maxv,maxi] = max(delta(:,index));
        %[normobslik, nsum] = normalise(obslik(:,t)); 
	%lik = lik + normobslik(maxi);
	lik = lik + (obslik(maxi,t));
        %normobslik,sum(normobslik)
	%break
  end
end
[p, path(T)] = max(delta(:,T));
for t=T-1:-1:1
  path(t) = psi(path(t+1),t+1);
end

nelem = (T-index);
%lik = lik/log(-(scale(index)));
% If scaled==0, p = prob_path(best_path)
% If scaled==1, p = Pr(replace sum with max and proceed as in the scaled forwards algo)
% Both are different from p(data) as computed using the sum-product (forwards) algorithm
if 0
if scaled 
  loglik = -sum(log(scale));
  %loglik = prob_path(prior, transmat, obslik, path);
else
  loglik = log(p);
end
end
