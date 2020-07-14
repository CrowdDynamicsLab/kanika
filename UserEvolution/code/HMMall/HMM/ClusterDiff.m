function [distance] = ClusterDiff(original, updated)

C = size(original,2);
S = size(original{1},2);

distance = 0.0;

for c = 1:C
    temp = zeros(C,1);
    for d = 1:C
        jsdiv = 0;
        for s = 1:S
            jsdiv = jsdiv + KLmultinomial(original{c}(:,s), updated{c}(:,s), 'js');
            
        end
        temp(d) = jsdiv;
    end
    dist = min(temp);
    distance = max(distance, dist);
end
