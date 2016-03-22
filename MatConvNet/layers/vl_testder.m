function [dzdx, dzdx_]=vl_testder(g,x,dzdy,dzdx,delta,tau)
% Utilities to test backward derivative computation against 
% numerical derivative of the forward pass. 
%
% This utilities is provided by MatConvNet

useGPU=false;

if nargin < 5
  delta = 1e-3 ;
end

if nargin < 6
  tau = [] ;
end

if ~useGPU
    dzdy = gather(dzdy) ;
    dzdx = gather(dzdx) ;

    y = gather(g(x)) ;
    dzdx_=zeros(size(dzdx));
else
    y=g(x);
    dzdx_=gpuArray.zeros(size(dzdx));
end

for i=1:numel(x)
    if mod(i, 100)==0
        fprintf('current i=%d\n',i)
    end
  x_ = x ;
  x_(i) = x_(i) + delta ;
  if ~useGPU
    y_ = gather(g(x_)) ;
  else
    y_ = g(x_);
  end
  factors = dzdy .* (y_ - y)/delta ;
  dzdx_(i) = dzdx_(i) + sum(factors(:)) ;
end
% yang debug
%vl_testsim(dzdx, dzdx_, tau);

