function [allhx, Ws] = mSDAhd(xfreq, xx, noise,layers, dd)
% xfreq: nxr input
% xx : Dxn input
% noise: corruption level
% layers: number of layers to stack
% dd: number of features in each partition

% allhx: (layers*d)xn stacked hidden representations

lambda = 1e-05;
[r, n] = size(xfreq);
[D, n] = size(xx);
disp('stacking hidden layers...')
allhx = [];
Ws={};
for layer = 1:layers
	disp(['layer:',num2str(layer)])
	tic
	if layer == 1
		newhx = zeros(r, n);
        	RNDIDX=randperm(D);
                for batch=1:ceil(D/dd)
                	idx=RNDIDX((batch-1)*dd+1:min(batch*dd, length(RNDIDX)));
			        hx = mDAhd(xfreq, xx(idx, :), noise, lambda);
			        newhx = newhx + hx;
                end
		newhx = newhx/ceil(D/dd);
		newhx = tanh(newhx);
	else
		[newhx, W] = mDA(prevhx, noise, lambda);
		Ws{layer} = W;
	end
	toc
	allhx = [allhx; newhx];
	prevhx = newhx;
end
