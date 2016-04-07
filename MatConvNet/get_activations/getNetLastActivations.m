function [activations, label_out]=getNetLastActivations...
                         (imdb, batchIds, netGpu, getBatchFunc, batchSize)
    isFirst=1;
    if nargin<5
        batchSize=128;
    end
    
    % if net is larger than 100k, the downsample it to 10k
    % mainly to be friendly to MIT Places dataset
    if numel(batchIds) > 100000
        batchIds=batchIds(randsample(numel(batchIds), 10000));
        fprintf(['Warning: In getNetLastActivations, input batch too large',...
                 ', downsampled to 10K']);
    end
    
    label_out=zeros(numel(batchIds),1);
    for i=1:ceil(numel(batchIds)/batchSize)
        fprintf('In getNetLastActivations, reading and get activations of batch %d\n',i);
        
        inter=getInter(i, batchSize, numel(batchIds));
        [curImages, labels]=getBatchFunc(imdb, batchIds(inter));
        % prefetch
        getBatchFunc(imdb, batchIds(getInter(i+1, batchSize, numel(batchIds))));
        
        label_out(inter)=labels;
        res=vl_simplenn(netGpu, gpuArray(curImages),[],[], 'conserveMemory', true);
        final_resp=gather(res(end).x);
        
        % assign the response to output structure
        if isFirst
            isFirst=0;
            dims=getDimOfBlob(final_resp);
            activations_out=zeros(dims{:}, numel(batchIds), 'single');
            fprintf('In getNetLastActivations, net output dimension is %d\n', dims{:});
        end
        if numel(dims)==1
            activations_out(:, inter)=final_resp;
        elseif numel(dims)==2
            % if an error occurs below, probably batchSize==1. It's not
            % supported except the last layer is fisher vector.
            activations_out(:,:, inter)=final_resp;
        elseif numel(dims)==3
            activations_out(:,:,:, inter)=final_resp;
        else
            error('too many dim')
        end
    end
    activations=squeeze(activations_out);
end

function inter=getInter(iseg, segLen, up)
    inter=(iseg-1)*segLen+1 : min(iseg*segLen, up);
end

function out=getDimOfBlob(resp)
    sz=size(resp);
    out=sz(1:(end-1));
    out=num2cell(out);
    if 0
        fprintf('Warning: special case for batchsize==1 \n');
        out=num2cell(sz);
    end
end
