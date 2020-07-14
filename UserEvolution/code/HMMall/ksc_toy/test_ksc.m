function test_ksc(ItemType)
% test ksc and plot cluster centroids
% ItemType = 1: Memetracker phrase
% ItemType = 2: Twitter Hashtags

if ItemType == 1
    fid = fopen('MemePhr.txt');
    rand('state', 2);
else
    fid = fopen('TwtHtag.txt');
    rand('state', 3);
end
    
% rand('state',0    
% close all;


X = [];
while 1
    tline = fgetl(fid);
    if(tline == -1) break; end;
    a = str2num(tline);
    if length(a)>0
        X = [X;a];
    end
end
fclose(fid);


b = X./repmat(max(X, [], 2),[1 size(X,2)]);
[ksc cent] = ksc_toy(X, 6);

figure;
for i=1:6
  subplot(2,3,i);
  plot(cent(i,:));
  title('ksc');
  axis([0 130 0 1.2 * max(cent(i,:))]);
end
