function out=consts(dataset, varName, varargin)
% All constants are set here, modify them if they don't fit on your system.
    opts.pretrainMethod = 'SVM';
    opts.network='VGG_M';
    opts.cfgId='bilinear';
    opts.projDim=5000;
    % use448 could be true/false, or could be an int32
    opts.use448=true;
    opts.learnW='false';
    opts.tag='no_tag';
    opts = vl_argparse(opts, varargin);
    if isa(opts.use448, 'logical')
        if opts.use448
            opts.use448=int32(448);
        else
            opts.use448=int32(224);
        end
    end

    expDir=fullfile('data','exp_yang_fv');
    
    % the expDirOne
    compactStr='';
    if strncmpi(opts.cfgId, 'compact', 7) || strcmp(opts.cfgId, 'pca_fb')
        if opts.learnW, lstr='learnW'; else lstr='fixW'; end
        compactStr=['_' num2str(opts.projDim) '_' lstr];
    end
    expDirOne=fullfile(expDir,...
      [dataset '_' opts.network '_' opts.pretrainMethod '_' opts.cfgId ...
       compactStr '_' num2str(opts.use448) '_' opts.tag]) ;
    % end of expDirOne
    
    VGG_M=fullfile('data','models','imagenet-vgg-m.mat');
    VGG_16=fullfile('data','models','imagenet-vgg-verydeep-16.mat');
    places_16=fullfile('data','models','places-vgg-16.mat');
    
    % some conditional configuration string
    % for the pca_fb case, there are two possible outputs:
    %   first, cfgId=="pcaConv"
    %   second,cfgId=="pca_fb". 
    % only the second case is dimensional specific, the first is not. 
    if strncmpi(opts.cfgId, 'compact', 7) || strcmp(opts.cfgId, 'pca_fb')
        strDim=['_' num2str(opts.projDim)];
    else
        strDim='';
    end
    strRes=['_' num2str(opts.use448)];
    
    classifierW=fullfile('data','exp_yang_fv', ...
        ['classifierW_' dataset '_' opts.network '_' opts.pretrainMethod '_' opts.cfgId ...
        strRes strDim '.mat']);
    
    % this is deprecated, use fisherInit instead
    initFVparamDeprecate=...
        fullfile('data','exp01',[ lower(dataset) '-seed-01'], 'dcnnfv-encoder.mat');
    fisherInit=fullfile('data','exp01',[ lower(dataset) '-seed-01'],...
        ['fisherInit_' opts.network strRes '.mat']);
    
    imdb=fullfile('data','exp01', [lower(dataset) '-seed-01'],'imdb','imdb-seed-1.mat'); 
    
    if strcmp(dataset, 'MIT')
        dataDir=fullfile('data','mit_indoor');
        num_classes=67;
    elseif strcmp(dataset, 'MIT_FULL')
        dataDir=fullfile('data','mit_full');
        num_classes=67;
    elseif strcmp(dataset, 'CUB') || ...
           strcmp(dataset, 'CUB_HEAD') || strcmp(dataset, 'CUB_BODY') || ...
           strcmp(dataset, 'CUB_HEAD_DETECT') || ...
           strcmp(dataset, 'CUB_BODY_DETECT')
        dataDir=fullfile('data', lower(dataset));
        num_classes=200;
    elseif strcmp(dataset, 'FMD')
        dataDir=fullfile('data', 'fmd');
        num_classes=10;
    elseif strcmp(dataset, 'DTD')
        dataDir=fullfile('data', 'dtd');
        num_classes=47;
    elseif strcmp(dataset, 'MIT_Places')
        dataDir=fullfile('data', 'mit_places');
        num_classes=201;
    else
        error('dataset unknown')
    end
    
    if strcmp(opts.network, 'VGG_M')
        endLayer=13;
        lastNchannel=512;
    elseif strcmp(opts.network, 'VGG_16')
        endLayer=29;
        lastNchannel=512;
    elseif strcmp(opts.network, 'places_16')
        endLayer=29;
        lastNchannel=512;
    else
        error('unknown network');
    end
    
    poolout=fullfile(expDir, ['poolout_', dataset, '_',...
        opts.network, '_', opts.cfgId, strRes strDim '.mat']);
    
    projInitW=fullfile('data','exp_yang_fv',...
        ['projectW_' opts.cfgId '_' num2str(lastNchannel) strDim '.mat']);
    
    pcaWeight=fullfile('data','exp_yang_fv',['pcaWeight_', dataset, '_', ...
               opts.network, strRes, '.mat']);
    
    eval(['out=' varName ';']);
end
