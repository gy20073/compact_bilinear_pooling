% IMPORTANT, this only implements signed square root normalization
% not L2 normalization, you have to use vl_nnnormalize() with param
% [N>=2D KAPPA=0 ALPHA=1 BETA=0.5] to get L2 normalization
% 
% Before each use, you should clear yang_fv

% TODO list: check again through all
% TODO(efficiency: 
%      picture subsample
%      before vl_gmm: number of each picture's local descriptors,
%      when extracting fisher vector: constraint local descriptors (also
%          change backward)) : see whether now works well??
% TODO(PCA support) : of little use
% TODO(Parallel & GPU) : not important
% TODO(retrain GMM has not tested)

% OUTSIDE TODO list
% TODO(multi-scale support)
% TODO(upsample (through geometric params) or downsample to cope with too 
%      small or too big images)
% TODO(minus average color)
% TODO(other SGD parameters, such as learning rate, batch size, etc. )

% DONE list
% 1. move GPU array back and forth (hasn't tested)
% 2. tested gradient calculation using numerical methods
% 3. update parameters before a forward (instead between forward & backward)

% comments
% 1. they pool over multiple scales using the same fv parameter. And they
% could miss some of the local descriptors if the image size does not fits.
% 2. error in q_{ik} on the webpage, but not in the code
function out=yang_fv(X, varargin)
% This function don't do any form of sanity check
% X is some input local features of size H x W x D x N
%
% parameters: num_mixture, samples_per_component, images_per_fv, init_param
% parameters are given in the initializations of the function, later values
% are ignored
%
% FORWARD case: Y=yang_fv(X)
% output fisher encoded Y of size: 1 x 1 x (2 x num_mixture x D) x N
% forward also saves a matrix of descriptors that's sampled from each image
% sample: Y=yang_fv(X, 'num_mixture', 64, 'samples_per_component', 1000,...
%                      'images_per_fv', 5000, 'init_param', param0);
%
% BACKWARD case: dzdx=yang_fv(X, dzdy, Y), where Y is the output of the
% corresponding input X (in the forward pass)
% size(dzdy)==size(Y), size(dzdx)==size(X)
% sample: dzdx=yang_fv(X, dzdy, Y);

    % param has fields: means(D*num_mixture), covariances(D*num_mixture),
    % priors(num_mixture)
    persistent param opts descr
    
    isBackward=0;
    if nargin==0
        error('No input parameters');
    elseif (nargin>=2) && (~ischar(varargin{1}))
        isBackward=1;
        dzdy=varargin{1};
        Yx=varargin{2};
        varargin=varargin(3:end);
    end
    
    % cope with gpu array
    onGPU=0;
    if isa(X, 'gpuArray') || (isBackward && (isa(dzdy, 'gpuArray') || isa(Yx, 'gpuArray')))
        onGPU=1;
        X=gather(X);
        if isBackward
            dzdy=gather(dzdy);
            Yx=gather(Yx);
        end
    end
    
    if isempty(opts)
        fprintf(['yang_fv: Initializing all internel persistent variables,', ...
                 'this should appear once and only once \n'])
        % the first call to this function has arrived, otherwise assume
        % param, opt are both initialized
        opts.num_mixture=64;
        opts.samples_per_component=1000;
        opts.images_per_fv=5000;
        opts.init_param=[];
        opts.disable_param_update=true;
        opts = vl_argparse(opts, varargin);
        
        param.means=opts.init_param{1};
        param.covariances=opts.init_param{2};
        param.priors=opts.init_param{3};
        
        opts=rmfield(opts, 'init_param');
        param.gmd=gmdistribution(param.means', ...
             reshape(param.covariances, 1 , size(param.covariances,1), []),...
             param.priors);
        
        opts.des_dim=size(param.means, 1);
        opts.fv_dim = 2 * opts.num_mixture * opts.des_dim;
        
        descr.total=opts.num_mixture * opts.samples_per_component;
        descr.each_image=ceil(descr.total / opts.images_per_fv);
        descr.cache_size=0; 
        descr.cache=zeros(opts.des_dim, descr.total, 'single');
    end
    
    % some short hands
    [xH, xW, xD, xN]=size(X);
    
    if isBackward
        % x is D*(H*W)
        dzdx=zeros(size(X),'single');
        dzdy=squeeze(dzdy);
        Yx=squeeze(Yx);
        for i=1:xN
            x=X(:,:,:,i);
            x=permute(x, [3 1 2]);
            x=reshape(x, xD, []);
            
            % tt is also D*(H*W)
            tt=backOne(x, dzdy(:,i), Yx(:,i), param);
            
            tt=reshape(tt, xD, xH, xW);
            tt=permute(tt, [2 3 1]);
            dzdx(:,:,:,i)=tt;
        end
        
        out=dzdx;
    else     % running forward
        if ~opts.disable_param_update
            % update param if possible
            if descr.cache_size==descr.total
                v = var(descr.cache, 2);
                [param.means, param.covariances, param.priors] = ...
                  vl_gmm(descr.cache, opts.num_mixture, ...
                        'InitMeans', param.means, ...
                        'InitCovariances', param.covariances, ...
                        'InitPriors', param.priors, ...
                        'CovarianceBound', double(max(v)*0.0001), ...
                        'NumRepetitions', 1);
                    % TODO(variance lower bound could be altered)
                descr.cache_size=0;

                param.gmd=gmdistribution(param.means', ...
                 reshape(param.covariances, 1 , size(param.covariances,1), []),...
                 param.priors);
            end
        end
        
        % calculate the forward output
        Y=zeros(opts.fv_dim, xN);
        for i=1:xN
            Y(:, i) = vl_fisher(reshape(X(:,:,:,i), [], xD)', ...
                                param.means, ...
                                param.covariances, ...
                                param.priors, ...
                                'SquareRoot');
                                % TODO(now could only implement square root)
                                %'Improved') ;
                                %TODO('Fast', could possibly improve speed)
            % Improved means: square root -> L2 normalize
        end
        
        if ~opts.disable_param_update
            % save the descriptor cahce
            tempX=permute(X, [1 2 4 3]); % of size H*W*N*D
            tempX=reshape(tempX, [], xD);
            % size(sel)==xD*(xN*each_image)
            hwn=size(tempX,1);
            sel=tempX(randsample(hwn, min(hwn, xN*descr.each_image)),:)'; 
            % in case the cache is full
            sel=sel(:, min(descr.total-descr.cache_size, size(sel, 2)));
            descr.cache(:, (descr.cache_size+1): ...
                           (descr.cache_size+size(sel,2)) )=sel;
            descr.cache_size=descr.cache_size+size(sel,2);
            clear tempX sel
        end
        
        % output
        out=reshape(Y,1,1,opts.fv_dim, xN);
    end
    
    out=single(out);
    % transfer back to GPU
    if onGPU
        out=gpuArray(out);
    end
end

function dzdx=backOne(x, uv, yx, param)
    % x is the local feature matrix for one image, size==D*N, where N==H*W
    % u is fv's first order statistic, size==D*K, similarly is v
    % mu is the mean matrix, size==D*K, similarly as cov
    mu=param.means;
    cov=param.covariances;
    % pi is the one dim prior, length K
    pi=param.priors;
    % q size==K*N, is the posterior probabilities
    q=posterior(param.gmd, x')';
    
    % some size short hands
    K=size(mu, 2);
    N=size(x, 2);
    D=size(x, 1);
    
    % for numerical stability
    epsilon=0.0001;
    yx = yx + epsilon;
    cov = cov + epsilon;
    pi = pi + epsilon;
    
    % DONE(invert the signed square root -> L2 normalize) from uv
    % just signed square root is used
    uv=uv./abs(yx)/2; % DONE(all '/' nees numerical stability concerns)
    % end of conversion
    u=reshape(uv(1:end/2), D, K);
    v=reshape(uv((end/2+1):end), D, K);
    
    % first term
    invSqrtCov=1./sqrt(cov); %DONE('/' check cov>epsilon)
    invSqrtPi=1./sqrt(pi(:))'; %DONE('/' check pi>epsilon)
    t1=(u-sqrt(2)*mu.*v.*invSqrtCov).*invSqrtCov; % size==D*K
    t1=bsxfun(@times, t1, invSqrtPi);
    t1=t1*q; % then size D*N
    
    % second term
    t2=v ./ cov;
    t2=bsxfun(@times, t2, invSqrtPi);
    t2=sqrt(2)* x .* (t2*q);
    
    % return
    dzdx=(t1+t2)./N;
end


