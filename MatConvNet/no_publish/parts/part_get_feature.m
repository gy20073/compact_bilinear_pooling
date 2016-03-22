function [trainFV, valFV]=part_get_feature(trainFV, valFV, imdb)
    parts_loc=imdb.images.parts_transformed;
    set=imdb.images.set;
    trainFV=get_one(trainFV, parts_loc(:,:,set==1));
    valFV=get_one(valFV, parts_loc(:,:,set==3));
end

function out=get_one(conv5, parts_loc)
    nparts=15;
    projDim=8192;
    neighbour=1; % is the delta 
    
    [h, w, c, n]=size(conv5);
    assert(h==w, 'Assumption that height==width failed');
    
    out=zeros(projDim*nparts, n, 'single');
    
    obj=yang_Compact_TS_2stream(projDim, [c c], 0);
    sq=@(x)sqrt(abs(x)) .* sign(x);
    %l2norm=@(x)single( bsxfun(@rdivide, x, sqrt(sum(x.^2, 1))) );
    l2norm=@(x) x ./ sqrt(sum(x(:).^2));
    
    nbatch=100;
    for iout=1:nbatch:n
        iout
        
        inter=iout:min(n, iout+nbatch-1);
        tt=gpuArray(conv5(:,:,:,inter));
        projected=obj.forward({tt, tt}, []);
        projected=gather(projected{1});
        
        for i=inter
            for j=1:nparts
                % if there exists that part
                if parts_loc(3, j, i)
                    getInter=@(px) max(1, px-neighbour): min(h, px+neighbour);
                    
                    pxy=parts_loc(1:2, j, i)/448*h;
                    px=round(pxy(1));
                    py=round(pxy(2));
                    ipx=getInter(px);
                    ipy=getInter(py);
                    if isempty(ipx) || isempty(ipy)
                        j 
                        i
                        continue
                    end

                    feature=sum(sum(...
                        projected(ipx, ipy, :, i-iout+1) ...
                                    ,1),2);
                    feature=squeeze(feature);
                    feature=l2norm(sq(feature));

                    out((projDim*(j-1)+1):projDim*j, i)=feature;
                end
            end
        end
        
    end
end