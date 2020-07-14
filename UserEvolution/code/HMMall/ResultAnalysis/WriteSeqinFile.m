K = 7; 
%f = fopen('DBLPData/TopPattern.txt','w');
f = fopen('StackExchangeData/TopPattern.txt','w');
format ='%s\n %s\n %s\n';
for k =1:K
fprintf(f,format,[mat2str(TopPattern{k,1}),mat2str(TopPattern{k,2}),mat2str(TopPattern{k,3})]);
end
% fprintf(f,format,[mat2str(TopPattern{2,1}),mat2str(TopPattern{2,2}),mat2str(TopPattern{2,3})]);
% fprintf(f,format,[mat2str(TopPattern{3,1}),mat2str(TopPattern{3,2}),mat2str(TopPattern{3,3})]);
% fprintf(f,format,[mat2str(TopPattern{4,1}),mat2str(TopPattern{4,2}),mat2str(TopPattern{4,3})]);
% fprintf(f,format,[mat2str(TopPattern{5,1}),mat2str(TopPattern{5,2}),mat2str(TopPattern{5,3})]);
% fprintf(f,format,[mat2str(TopPattern{6,1}),mat2str(TopPattern{6,2}),mat2str(TopPattern{6,3})]);
fclose(f);

%f = fopen('DBLPData/BottomPattern.txt','w');
f = fopen('StackExchangeData/BottomPattern.txt','w');

format ='%s\n %s\n %s\n';
for k = 1:K
fprintf(f,format,[mat2str(BottomPattern{k,1}),mat2str(BottomPattern{k,2}),mat2str(BottomPattern{k,3})]);
end
% fprintf(f,format,[mat2str(BottomPattern{2,1}),mat2str(BottomPattern{2,2}),mat2str(BottomPattern{2,3})]);
% fprintf(f,format,[mat2str(BottomPattern{3,1}),mat2str(BottomPattern{3,2}),mat2str(BottomPattern{3,3})]);
% fprintf(f,format,[mat2str(BottomPattern{4,1}),mat2str(BottomPattern{4,2}),mat2str(BottomPattern{4,3})]);
% fprintf(f,format,[mat2str(BottomPattern{5,1}),mat2str(BottomPattern{5,2}),mat2str(BottomPattern{5,3})]);
% fprintf(f,format,[mat2str(BottomPattern{6,1}),mat2str(BottomPattern{6,2}),mat2str(BottomPattern{6,3})]);
fclose(f);

%f = fopen('DBLPData/MidPattern.txt','w');
f = fopen('StackExchangeData/MidPattern.txt','w');
format ='%s\n %s\n %s\n';
for k = 1:K
fprintf(f,format,[mat2str(MiddlePattern{k,1}),mat2str(MiddlePattern{k,2}),mat2str(MiddlePattern{k,3})]);
end
fclose(f);

%f = fopen('DBLPData/FreqPattern.txt','w');
f = fopen('StackExchangeData/FreqPattern.txt','w');
format ='%s\n %s\n %s\n';
for k =1:K
    fprintf(f,format,mat2str(FrequentSequence(k,:)));
end
% fprintf(f,format,mat2str(FrequentSequence(2,:)));
% fprintf(f,format,mat2str(FrequentSequence(3,:)));
% fprintf(f,format,mat2str(FrequentSequence(4,:)));
% fprintf(f,format,mat2str(FrequentSequence(5,:)));
%fprintf(f,format,mat2str(FrequentSequence(6,:)));
fclose(f);

%f = fopen('DBLPData/CompactFreqPattern.txt','w');
f = fopen('StackExchangeData/CompactFreqPattern.txt','w');
format ='%s\n %s\n %s\n';
for k =1:K
    fprintf(f,format,mat2str(FrequentCompactSeq(k,:)));
    fprintf(f,format,mat2str(WeightFrequentCompactSeq(k,:)));
end
fclose(f);
