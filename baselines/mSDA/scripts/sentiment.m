% an example using mSDA to generate features for sentiment analysis on the Amazon review dataset of (Blitzer et al., 2006), using only the top 5,000 features
addpath('../lowDimen')
addpath('./libsvm-3.11/matlab/');
domains=cell(4,1);
domains{1}='books';
domains{2}='dvd';
domains{3}='electronics';
domains{4}='kitchen';

disp(['mSDA on amazon'])

folds = 5;
% two hyper-parameters to be cross-validated
% number of mSDA layers to be stacked
layers=5;
% corruption level
noises=[0.5,0.6,0.7,0.8,0.9];

% read in the raw input
load('amazon.mat');
dimen = 5000;
xx = xx(1:dimen, :);

ACCs=zeros(length(noises), size(domains,1));
Cs=zeros(length(noises), size(domains,1));

% cross validate on the corruption level
for iter = 1:length(noises)
	noise=noises(iter);
	disp(['corruption level ', num2str(noise)])
	% generate hidden representations using mSDA
	[allhx] = mSDA(double(xx>0),noise,layers);
        for j = 1:size(domains,1)
		source=domains{j};
		disp(['domain ',source, ' ...'])
		yr=yy(offset(j)+1:offset(j)+2000);
		xr=[xx(:, offset(j)+1:offset(j)+2000); allhx(:, offset(j)+1:offset(j)+2000)];
		xr=xr';
		Cs(iter, j) = 1./mean(sum(xr.*xr,2));
		model = svmtrain(yr,xr,['-q -t 0 -c ',num2str(Cs(iter,j)),' -m 3000']);
		ACCs(iter, j) = svmtrain(yr,xr,['-q -t 0 -c ',num2str(Cs(iter,j)),' -v ', num2str(folds), ' -m 3000']);
        end
	fprintf('\n')
end

% finalize training and testing
[temp, noiseIdx]=max(ACCs);
for j = 1:size(domains,1)
	source=domains{j};
	yr=yy(offset(j)+1:offset(j)+2000);
	bestNoise = noises(noiseIdx(j));
	disp(['learn representation with corruption level ' num2str(bestNoise), ' ...']);
	[allhx] = mSDA(double(xx>0), bestNoise, layers);
	xr=[xx(:, offset(j)+1:offset(j)+2000); allhx(:, offset(j)+1:offset(j)+2000)];
	xr=xr';
	bestC=Cs(noiseIdx(j),j);
	disp(['final training on domain ', source, ' ...'])
	model = svmtrain(yr,xr,['-q -t 0 -c ',num2str(bestC),' -m 3000']);
	for i = 1:size(domains,1)
		target=domains{i};
		if i == j
			continue;
		end
		disp(['final testing on domain ', target, ' ...'])
		xe=[xx(:, offset(i)+2001:offset(i+1)); allhx(:, offset(i)+2001:offset(i+1))];
		xe=xe';
		ye=yy(offset(i)+2001:offset(i+1));
		[pred,accuracy] = svmpredict(ye,xe,model);

		ev=metrics(ye,pred);
		disp(['precision: ',num2str(ev(2))]);
		disp(['recall: ',num2str(ev(3))]);
		disp(['f1: ',num2str(ev(4))]);

		% Print to file
		disp(['printing predictions to file...'])
		fname=strcat('results/mSDA','-',source, '-', target);
		fileID=fopen(fname,'w');
		fprintf(fileID,'%d\n',pred);
		fclose(fileID);

		% Print gold to file
		fname=strcat('results/', target, '-gold.txt');
		fileID=fopen(fname,'w');
		fprintf(fileID,'%d\n',ye);
		fclose(fileID);

	end
	fprintf('\n');
end



