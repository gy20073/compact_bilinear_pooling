function out=getBatchFromDisk(imdb, diskFile)
    % !!! currently only supports my disk file format
    % i.e. the four variables exists in the file
    
    % We use global variable to avoid multiple load from disk files, this
    % is not an elegant solution
    global trainFV trainY valFV valY mapGlobal2local
    load(diskFile);
    % !!! also assume that each query only in train or validation
    
    % now build the map that maps global id to train/val ids
    ntrain=size(trainFV, 2);
    ntest=size(valFV, 2);
    mapGlobal2local=zeros(1, ntrain+ntest, 'single');
    iset=imdb.images.set;
    mapGlobal2local(iset==1)= (1:ntrain);
    mapGlobal2local(iset~=1)= (1:ntest);
    % now mapGlobal2local(ids) should map to local ids
    trainFV=reshape(trainFV, 1,1, [], ntrain);
    valFV=reshape(valFV, 1,1,[], ntest);

    out=@(imdb, batch) impl(imdb, batch, ...
        trainFV, trainY, valFV, valY, mapGlobal2local);
end

function [im, labels]=impl(imdb, batch, trainFV, trainY, valFV, valY, mapGlobal2local)
    if isempty(batch)
        im=[];
        labels=[];
        return;
    end
    iset=imdb.images.set;
    curSet=iset(batch)-1;
    % assert all in the same set
    assert(all(curSet) || all(~curSet));
    
    newIds=mapGlobal2local(batch);
    if curSet(1)
        % in the validation set
        im=valFV(1,1,:, newIds);
        labels=valY(newIds);
    else
        % in the training set
        im=trainFV(1,1,:,newIds);
        labels=trainY(newIds);
    end
end
