function y = vl_nnbilinearpool(x, varargin)
% VL_NNBILINEARPOOL pools bilinear feature across all locations

% Copyright (C) 2015 Tsung-Yu Lin, Aruni RoyChowdhury, Subhransu Maji.
% All rights reserved.
%
% This file is part of the BCNN and is made available under
% the terms of the BSD license (see the COPYING file).


% flag for doing backward pass
backMode = numel(varargin) > 0 && ~isstr(varargin{1}) ;
if backMode
  dzdy = varargin{1};
end

% if GPU is used
gpuMode = isa(x, 'gpuArray');

% [height width channels batchsize]
[h, w, ch, bs] = size(x);

if backMode
    % do backward pass
    if gpuMode
        y = gpuArray(zeros(size(x), 'single'));
    else
        y = zeros(size(x), 'single');
    end
    for b=1:bs
        dzdy_b = reshape(dzdy(1,1,:,b), [ch, ch]);
        a = reshape(x(:,:,:,b), [h*w, ch]);
        % yang: bug here, should be 2* the original
        y(:, :, :, b) = 2*reshape(a*dzdy_b, [h, w, ch]);
    end
else
    % do forward pass
    if gpuMode
        y = gpuArray(zeros([1, 1, ch*ch, bs], 'single'));
    else
        y = zeros([1, 1, ch*ch, bs], 'single');
    end
    for b = 1:bs,
        a = reshape(x(:,:,:,b), [h*w, ch]);
        y(1,1,:, b) = reshape(a'*a, [1 ch*ch]);
    end
end
