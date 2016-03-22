function vl_simplenn_diagnose(net, res)
% VL_SIMPLENN_DIAGNOSE  Plot diagnostic information
%   VL_SIMPLENN_DIAGNOSE(NET, RES) plots in the current window
%   the average, maximum, and miminum element for all the filters
%   and biases in the network NET. If RES is also provided, it will
%   plot the average, minimum, and maximum element for all the
%   intermediate responses and deriviatives stored in RES as well.
%
%   This function can be used to rapidly glance at the evolution
%   of the paramters during training.
%%
n = numel(net.layers) ;
fmu = NaN + gpuArray.zeros(1, n) ;
fmi = fmu ;
fmx = fmu ;
bmu = fmu ;
bmi = fmu ;
bmx = fmu ;
xmu = fmu ;
xmi = fmi ;
xmx = fmx ;
dxmu = fmu ;
dxmi = fmi ;
dxmx = fmx ;
dfmu = fmu ;
dfmi = fmu ;
dfmx = fmu ;
dbmu = fmu ;
dbmi = fmu ;
dbmx = fmu ;

for i=1:numel(net.layers)
    if i==16 || i==17
        %continue
    end
  ly = net.layers{i} ;
  if isfield(ly, 'weights') && numel(ly.weights{1}) > 0
    %x = gather(ly.weights{1}) ;
    x = ly.weights{1};
    [fmu(i), fmx(i), fmi(i)]=getSt(x);
  end
  if  isfield(ly, 'weights') && numel(ly.weights{2}) > 0
    %x = gather(ly.weights{2}) ;
    x = ly.weights{2};
    [bmu(i), bmx(i), bmi(i)]=getSt(x);
  end
  if nargin > 1
    if numel(res(i).x) > 1
      %x = gather(res(i).x) ;
      x = res(i).x;
      [xmu(i), xmx(i), xmi(i)]=getSt(x);
    end
    if numel(res(i).dzdx) > 1
      x = res(i).dzdx;
      [dxmu(i), dxmx(i), dxmi(i)]=getSt(x);
    end
    if isfield(res(i), 'dzdw') && numel(res(i).dzdw)>=2
        if isfield(ly, 'weights') && numel(res(i).dzdw{1}) > 0
          x = res(i).dzdw{1};
          [dfmu(i), dfmx(i), dfmi(i)]=getSt(x);
        end
        if isfield(ly, 'weights') && numel(res(i).dzdw{2}) > 0
          x = res(i).dzdw{2};
          [dbmu(i), dbmx(i), dbmi(i)]=getSt(x);
        end
    end
  end
end

if nargin > 1
  np = 6 ;
else
  np = 2 ;
end

clf ; subplot(np,1,1) ;
errorbar(1:n, fmu, fmi, fmx, 'bo') ;
grid on ;
xlabel('layer') ;
ylabel('filters') ;
title('coefficient ranges') ;

subplot(np,1,2) ;
errorbar(1:n, bmu, bmi, bmx, 'bo') ;
grid on ;
xlabel('layer') ;
ylabel('biases') ;

if nargin > 1
  subplot(np,1,3) ;
  errorbar(1:n, xmu, xmi, xmx, 'bo') ;
  grid on ;
  xlabel('layer') ;
  ylabel('x') ;

  subplot(np,1,4) ;
  errorbar(1:n, dxmu, dxmi, dxmx, 'bo') ;
  grid on ;
  xlabel('layer') ;
  ylabel('dzdx') ;

  subplot(np,1,5) ;
  errorbar(1:n, dfmu, dfmi, dfmx, 'bo') ;
  grid on ;
  xlabel('layer') ;
  ylabel('dfilters') ;

  subplot(np,1,6) ;
  errorbar(1:n, dbmu, dbmi, dbmx, 'bo') ;
  grid on ;
  xlabel('layer') ;
  ylabel('dbiases') ;
end


drawnow ;
end

function [m, above, below]=getSt(x)
    m=mean(x(:));
    above=max(x(:))-m;
    below=m-min(x(:));
end
