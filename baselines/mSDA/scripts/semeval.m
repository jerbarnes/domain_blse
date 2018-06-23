% an example using mSDA to generate features for sentiment analysis on the Amazon review dataset of (Blitzer et al., 2006), using only the top 5,0
addpath('../highDimen')
addpath('./libsvm-3.11/matlab/');
folds = 5;
% two hyper-parameters to be cross-validated
% number of mSDA layers to be stacked
layers=5;
% corruption level
noises=[0.5,0.6,0.7,0.8,0.9];

% read in the raw input
load('mine.mat');
load('semeval2013.mat')
y = double(yy');
y_trg = double(y_trg');

dimen = 30000;
x = [xx,x_trg];
x = x(1:dimen, :);

% dd: the number of features in each partition
dd = 5000;;
freqIdx = 1:5000;
xfreq = x(freqIdx, :);
xfreq = double(xfreq>0);

ACCs=zeros(length(noises), 1);
Cs=zeros(length(noises), 1);

% cross validate on the corruption level for amazon
for iter = 1:length(noises)
	noise=noises(iter);
	disp(['corruption level ', num2str(noise)])
	% generate hidden representations using mSDA
	[allhx] = mSDAhd(xfreq, double(x>0), noise,layers, dd);
	xr=[x(:, 1:2000); allhx(:, 1:2000)];
	xr=xr';
    Cs(iter) = 1./mean(sum(xr.*xr,2));
    disp(['training model...'])
	model = svmtrain(y,xr,['-q -t 0 -c ',num2str(Cs(iter)),' -m 3000']);
	ACCs(iter) = svmtrain(y,xr,['-q -t 0 -c ',num2str(Cs(iter)),' -v ', num2str(folds), ' -m 3000']);
	fprintf('\n')
end

% finalize training and testing
[temp, noiseIdx]=max(ACCs);
bestNoise = noises(noiseIdx);
disp(['learn representation with corruption level ' num2str(bestNoise), ' ...']);
[allhx] = mSDAhd(xfreq, double(x>0), bestNoise, layers, dd);
xr=[x(:, 1:2000); allhx(:, 1:2000)];
xr=xr';
bestC=Cs(noiseIdx);
disp(['final training on amazon ...'])
model = svmtrain(y,xr,['-q -t 0 -c ',num2str(bestC),' -m 3000']);
disp(['final testing on domain semeval ...'])
xe=[x(:, 2001:5056); allhx(:, 2001:5056)];
xe=xe';
ye=y_trg;
[label,accuracy] = svmpredict(ye,xe,model);
ev=metrics(ye,label);
disp(['precision: ',num2str(ev(2))]);
disp(['recall: ',num2str(ev(3))]);
disp(['f1: ',num2str(ev(4))]);