addpath('../highDimen')
addpath('./libsvm-3.11/matlab/');
domains=cell(5,1);
domains{1}='books';
domains{2}='dvd';
domains{3}='electronics';
domains{4}='kitchen';
domains{5}='all';

folds = 5;
% two hyper-parameters to be cross-validated
% number of mSDA layers to be stacked
layers=5;
% corruption level
noises=[0.5,0.6,0.7,0.8,0.9];

% read in the raw input
load('amazon+semeval.mat');


y = [double(yy);double(trg_y);double(trg_y2)];
trg_y = double(trg_y);
trg_y2 = double(trg_y2);

dimen = 30000;
x = [xx,trg_x,trg_x2];
x = x(1:dimen, :);

% dd: the number of features in each partition
dd = 5000;;
freqIdx = 1:5000;
xfreq = x(freqIdx, :);
xfreq = double(xfreq>0);

ACCs=zeros(length(noises), size(domains,1));
Cs=zeros(length(noises), size(domains,1));

% cross validate on the corruption level for amazon
for iter = 1:length(noises)
	noise=noises(iter);
	disp(['corruption level ', num2str(noise)])
	% generate hidden representations using mSDA
	[allhx] = mSDAhd(xfreq, double(x>0), noise,layers, dd);
	for j = 1:size(domains,1)
		source=domains{j};
		disp(['domain ',source, ' ...'])
		if j == 5
			yr=y(1:8000);
			xr=[x(1:8000); allhx(1:8000)];
			xr=xr';
		else
			yr=y(offset(j)+1:offset(j)+2000);
			xr=[x(:, offset(j)+1:offset(j)+2000); allhx(:, offset(j)+1:offset(j)+2000)];
			xr=xr';
		end
		Cs(iter, j) = 1./mean(sum(xr.*xr,2));
		disp(['training model...'])
		model = svmtrain(yr,xr,['-q -t 0 -c ',num2str(Cs(iter,j)),' -m 3000']);
		ACCs(iter, j) = svmtrain(yr,xr,['-q -t 0 -c ',num2str(Cs(iter,j)),' -v ', num2str(folds), ' -m 3000']);
    end
	fprintf('\n')
end

% train on each domain and test on semeval
[temp, noiseIdx]=max(ACCs);
for j = 1:size(domains,1)
	source=domains{j};
	yr=y(offset(j)+1:offset(j)+2000);
	bestNoise = noises(noiseIdx(j));

	% Learn representation
	disp(['learn representation with corruption level ' num2str(bestNoise), ' ...']);
	[allhx] = mSDAhd(xfreq, double(x>0), bestNoise, layers, dd);
	if j == 5
		yr=y(1:8000);
		xr=[x(:, 1:8000); allhx(:, 1:8000)];
		xr=xr';
	else
		yr=y(offset(j)+1:offset(j)+2000);
		xr=[x(:, offset(j)+1:offset(j)+2000); allhx(:, offset(j)+1:offset(j)+2000)];
		xr=xr';
	end

	% Train on best C for that dataset
	bestC=Cs(noiseIdx(j),j);
	disp(['final training on ', source, ' ...'])
	model = svmtrain(yr,xr,['-q -t 0 -c ',num2str(bestC),' -m 3000']);

	% test on semeval 2013
	disp(['final testing on semeval 2013...'])
	xe=[x(:, offset(5)+1:offset(6)); allhx(:, offset(5)+1:offset(6))];
	xe=xe';
	ye=trg_y;
	[pred,accuracy] = svmpredict(ye,xe,model);
	ev=metrics(ye,pred);
	disp(['precision: ',num2str(ev(2))]);
	disp(['recall: ',num2str(ev(3))]);
	disp(['f1: ',num2str(ev(4))]);

	% Print to file
	disp(['printing predictions to file...'])
	fname=strcat('results/mSDA','-',source, '-semeval_2013.txt');
	fileID=fopen(fname,'w');
	fprintf(fileID,'%d\n',pred);
	fclose(fileID);

	% Print gold to file
	fname=strcat('results/semeval_2013.gold.txt');
	fileID=fopen(fname,'w');
	fprintf(fileID,'%d\n',trg_y);
	fclose(fileID);

	% test on semeval 2016
	disp(['final testing on semeval 2016...'])
	xe=[x(:, offset(6)+1:offset(7)); allhx(:, offset(6)+1:offset(7))];
	xe=xe';
	ye=trg_y2;
	[pred,accuracy] = svmpredict(ye,xe,model);
	ev=metrics(ye,pred);
	disp(['precision: ',num2str(ev(2))]);
	disp(['recall: ',num2str(ev(3))]);
	disp(['f1: ',num2str(ev(4))]);

	% Print to file
	disp(['printing predictions to file...'])
	fname=strcat('results/mSDA','-',source, '-semeval_2016.txt');
	fileID=fopen(fname,'w');
	fprintf(fileID,'%d\n',pred);
	fclose(fileID);

	% Print gold to file
	fname=strcat('results/semeval_2016.gold.txt');
	fileID=fopen(fname,'w');
	fprintf(fileID,'%d\n',trg_y2);
	fclose(fileID);
end

