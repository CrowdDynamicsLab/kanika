function [KL] = KLmultinomial(pVect1,pVect2,varargin)

% Check probabilities sum to 1:
if ((abs(sum(pVect1) - 1) > .05) || (abs(sum(pVect2) - 1) > .05)) && sum(pVect1)~=0,
    pVect1, pVect2
    error('Probablities don''t sum to 1.')
end

pVect1 = pVect1+0.00001;
pVect2 = pVect2+0.00001;
pVect1 = pVect1/norm(pVect1,1);
pVect2 = pVect2/norm(pVect2,1);

if ~isempty(varargin),
    switch varargin{1},
        case 'js',
            logQvect = log2((pVect2+pVect1)/2);
            KL = .5 * (sum(pVect1.*(log2(pVect1)-logQvect)) + ...
                sum(pVect2.*(log2(pVect2)-logQvect)));

        case 'sym',
            KL1 = sum(pVect1 .* (log2(pVect1)-log2(pVect2)));
            KL2 = sum(pVect2 .* (log2(pVect2)-log2(pVect1)));
            KL = (KL1+KL2)/2;
            
        otherwise
            error(['Last argument' ' "' varargin{1} '" ' 'not recognized.'])
    end
else
    KL = sum(pVect1 .* (log2(pVect1)-log2(pVect2)));
end






