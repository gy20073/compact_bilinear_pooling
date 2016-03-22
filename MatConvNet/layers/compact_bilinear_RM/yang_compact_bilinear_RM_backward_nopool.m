function pre=yang_compact_bilinear_RM_backward_nopool(layer, pre, now)
    ws=layer.weights;
    learnW=layer.learnW; 
    %% setup the output variables
    pre.dzdx=[];
    [h,w,c,n]=size(pre.x);
    out=zeros([h*w, c, n], 'like', now.dzdx);

    if learnW    
        dzdw=zeros(size(ws{1}), 'like', now.dzdx);
        pre.dzdw={dzdw, dzdw};
    end
    
    % faster block calculation
    xin=permute(pre.x, [1,2,4,3]); % size h,w,n,c
    xin=reshape(xin, h*w*n, c);
    
    dzdy=permute(now.dzdx, [1 2 4 3]); % size h w n outDim
    dzdy=reshape(dzdy, h*w*n, size(ws{1}, 2));

    for iorder=1:2
        wi=ws{iorder};
        wip=ws{3-iorder};
        tin=(xin*wi).*dzdy; % tin is hwn * outDim
        if learnW
            pre.dzdw{3-iorder}=xin'*tin;
        end
        tin=tin*wip';
        tin=reshape(tin, h*w, n, c);
        out=out+permute(tin, [1,3,2]);
    end
 
%   slower loop calculation that is abandoned.  
%   THIS IS THE POOLED VERSION!!!   
%     %% calculate dzdx (and dzdw)
%     for img=1:n
%         % image size is [s, ch]
%         cur=reshape(pre.x(:,:,:,img), h*w, c);
%         der=reshape(now.dzdx(:,:,:,img), 1, []); % size [1, outDim] 
%         for iorder=1:2
%             weight=ws{iorder}; % size [ch, outDim]
%             wprime=ws{3-iorder}; % same size as above
%             
%             % computation = hw*c*outDim
%             wix=cur*weight; % size [s, outDim]
%             % add the dL/dy into wix
%             wix=bsxfun(@times, wix, der); % same size as wix
%             
%             % computation = hw*outDim*c
%             out(:,:,img)=out(:,:,img) + wix*wprime';
%             
%             % support for learning the parameters
%             if learnW
%                 pre.dzdw{3-iorder} = pre.dzdw{3-iorder}+cur'*wix;
%             end
%         end   
%     end
    
    pre.dzdx=reshape(out, h, w, c, n);
end
