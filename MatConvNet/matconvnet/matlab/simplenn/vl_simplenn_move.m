function net = vl_simplenn_move(net, destination)
% VL_SIMPLENN_MOVE  Move a simple CNN between CPU and GPU
%    NET = VL_SIMPLENN_MOVE(NET, 'gpu') moves the network
%    on the current GPU device.
%
%    NET = VL_SIMPLENN_MOVE(NET, 'cpu') moves the network
%    on the CPU.

% Copyright (C) 2014 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
    function x=testNan(x)
      if any(isnan(x(:)))
          error('nan in net');
      end
    end

switch destination
  case 'gpu', moveop = @(x) gpuArray(testNan(x)); 
  case 'cpu', moveop = @(x) gather(testNan(x)) ;
  otherwise, error('Unknown destination ''%s''.', destination) ;
end
for l=1:numel(net.layers)
  switch net.layers{l}.type
    case {'conv', 'convt', 'bnorm', 'custom'}
      for f = {'filters', 'biases', 'filtersMomentum', 'biasesMomentum'}
        f = char(f) ;
        if isfield(net.layers{l}, f)
          net.layers{l}.(f) = moveop(net.layers{l}.(f)) ;
        end
      end
      for f = {'weights', 'momentum'}
        f = char(f) ;
        if isfield(net.layers{l}, f)
          for j=1:numel(net.layers{l}.(f))
            net.layers{l}.(f){j} = moveop(net.layers{l}.(f){j}) ;
          end
        end
      end
    otherwise
      % nothing to do ?
  end
  
  for f ={'h', 's', 'outDim'}
      f=char(f);
      if isfield(net.layers{l}, f)
          net.layers{l}.(f) = moveop(net.layers{l}.(f));
          fprintf('moving %s\n', f);
      end
  end
  
   for f ={'comp','ia'}
      f=char(f);
      if isfield(net.layers{l}, f)
          for j=1:numel(net.layers{l}.(f))
            net.layers{l}.(f){j} = moveop(net.layers{l}.(f){j});
            fprintf('moving %s\n', f);
          end
      end
   end
  
   if isfield(net.layers{l}, 'layerObj')
       switch destination
           case 'gpu', net.layers{l}.layerObj.move2GPU(gpuArray(1));
           case 'cpu', net.layers{l}.layerObj.move2CPU();
       end
   end
end


end