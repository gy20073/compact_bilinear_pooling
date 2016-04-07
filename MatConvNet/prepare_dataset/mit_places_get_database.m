function imdb = mit_places_get_database(datasetDir)
    %%
    imdb.imageDir = fullfile(datasetDir, 'images256') ;
    imdb.sets = {'train', 'val', 'test'} ;
    % build 
    % imdb.classes.name: cell of #classes
    % imdb.images.name: cell of #images, paths
    % imdb.images.label: array of numerical labels

    nextCat=1;
    % top level : initials
    topLevel = dir(imdb.imageDir);
    for initial = 1 : numel(topLevel)
        curDir = topLevel(initial);
        if curDir.isdir && ~ismember({curDir.name}, {'.','..'})
            curDir = fullfile(imdb.imageDir, curDir.name);

            % second level : categories
            catDirs = dir(curDir);
            for cati = 1 : numel(catDirs)
                catDir = catDirs(cati);
                if catDir.isdir && ~ismember({catDir.name}, {'.','..'})
                    % imdb.classes.name
                    imdb.classes.name{nextCat} = catDir.name;
                    fprintf('extracting cat: %s \n', catDir.name);

                    % imdb.images.name/label
                    ims = dir(fullfile(curDir, catDir.name, '*.jpg'));
                    imdb.images.name{nextCat} = ...
                        cellfun(@(S) fullfile(topLevel(initial).name ,catDir.name, S), ...
                                {ims.name}, 'Uniform', 0);
                    imdb.images.label{nextCat} = nextCat * ones(1,numel(ims)) ;

                    nextCat = nextCat + 1;
                end
            end

        end
    end

    % turn the cell arrays to 1-dim list
    imdb.images.name = horzcat(imdb.images.name{:}) ;
    imdb.images.label = horzcat(imdb.images.label{:}) ;

    % generate the rest
    nimages=numel(imdb.images.name);
    imdb.images.id = 1:nimages;
    % set, reserve 1/10 for testing
    imdb.images.set = ones(1, nimages);
    imdb.images.set(1 : 10 : end) = 3;
end

