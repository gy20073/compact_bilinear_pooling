function prepare_dataset(dataset)
    mkdir(consts(dataset, 'expDir'));

    dataDir=consts(dataset, 'dataDir');
    if ~exist(dataDir, 'dir')
        error('Please download and unzip the %s dataset into %s folder', dataset, dataDir);
    end
    
    imdbFile=consts(dataset, 'imdb');
    if exist(imdbFile, 'file')
        % then we have already built the imdb structure
        return;
    end
   
    % dataset could be: MIT, CUB, FMD, DTD
    if strcmp(dataset, 'CUB')
        imdb=cub_get_database(dataDir);
    elseif strcmp(dataset, 'MIT')
        imdb=mit_indoor_get_database(dataDir, 'seed', 1);
    elseif strcmp(dataset, 'FMD')
        imdb=fmd_get_database(dataDir, 'seed', 1);
    elseif strcmp(dataset, 'DTD');
        imdb=dtd_get_database(dataDir, 'seed', 1);
    elseif strcmp(dataset, 'MIT_Places')
        imdb=mit_places_get_database(dataDir);
    else
        error('unknown dataset');
    end
    
    imdbDir=fileparts(imdbFile);
    mkdir(imdbDir);
    save(imdbFile, '-struct', 'imdb') ;
end
