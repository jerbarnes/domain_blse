function [hx, W] = mDAhd(xfreq,xx,noise,lambda)
% xfreq: rxn input
% xx : dxn input
% noise: corruption level
% lambda: regularization

% hx: dxn hidden representation
% W: dx(d+1) mapping

[r, n] = size(xfreq);
[d, n] = size(xx);
% adding bias
xxb = [xx; ones(1, n)];

% scatter matrix S
S = xxb*xxb';

% corruption vector
q = ones(d+1, 1)*(1-noise);
q(end) = 1;

% Q: (d+1)x(d+1)
Q = S.*(q*q');
Q(1:d+2:end) = q.*diag(S);

% P: rx(d+1)
P = xfreq*xxb'.*repmat(q', r, 1);

% final W = P*Q^-1, dx(d+1);
reg = lambda*eye(d+1);
reg(end,end) = 0;
W = P/(Q+reg);

hx = W*xxb;
