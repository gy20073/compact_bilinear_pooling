% explore the effect of different kernels on the orderless pooling problem
% visualize the kernels
function [bl, chi2, exp_chi2, chi_sq]=explore_kernels(trainFV)
    % sample two images and output their indices
    inds=randsample(size(trainFV, 4),2)
    
    a=trainFV(:,:,:,inds(1));
    b=trainFV(:,:,:,inds(2));
    
    bl=pair(a, b, 'bilinear');
    chi2=pair(a, b, 'chi2');
    exp_chi2=pair(a, b, 'exp_chi2'); 
    chi_sq=pair(a.^2, b.^2, 'chi2');
    chi_sq4=pair(a.^4, b.^4, 'chi2');
    
    vis={bl, chi2, exp_chi2, chi_sq, chi_sq4};
    names={'bilinear', 'chi2', 'exp-chi2','chi2-square', 'chi2-pow4'};
    
    nvis=numel(vis);
    for i=1:nvis
        subplot(2, nvis, i);
        imshow(vis{i});
        title(['similarity map for ' names{i}]);
        subplot(2, nvis, i+nvis);
        hist(vis{i}(:), 100);
        title(['histogram map for ' names{i}]);
    end
end

% input to pair is two image blobs, output a 169*169 matrix of similarity
% numbers
function out=pair(a, b, name)
    bilinear=@(x, y, gamma)dot(x, y).^2;
    chi2=@(x, y, gamma)sum(x.*y ./(x+y+1e-10));
    exp_chi2=@(x, y, gamma) exp(-gamma*chi2(x, y, 1));
    eval(['method=' name ';']);
    
    if strcmp(name, 'chi2') || strcmp(name, 'exp_chi2')
        [a, b]=normalize_l1_instance(a, b);
        a=squeeze(a);
        b=squeeze(b);
    end

    [h,w,c]=size(a);
    hw=h*w;
    out=zeros(hw, hw);
    a=reshape(a, hw, c);
    b=reshape(b, hw, c);
    for i=1:hw
        for j=1:hw
            out(i,j)=method(a(i,:), b(j,:), 1);
        end
    end
end

% abandoned, because the equavilance is verified. 
% total input two blobs, output the similarity scores by pooling first
function out=total_bilinear(a, b)
    [h,w,c]=size(a);
    hw=h*w;
    a=reshape(a, hw, c);
    b=reshape(b, hw, c);
    ta=zeros(c,c);
    tb=zeros(c,c);
    for i=1:hw
        ta=ta+a(i,:)'*a(i,:);
        tb=tb+b(i,:)'*b(i,:);
    end
    out=dot(ta(:), tb(:));
end

