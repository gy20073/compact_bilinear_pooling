function imdb = mit_indoor_get_database(datasetDir, varargin)
opts.seed = 1 ;
opts = vl_argparse(opts, varargin) ;

rng(opts.seed) ;

imdb.imageDir = fullfile(datasetDir, 'images') ;
imdb.maskDir = fullfile(datasetDir, 'mask') ;

cats = dir(imdb.imageDir) ;
cats = cats([cats.isdir] & ~ismember({cats.name}, {'.','..'})) ;
imdb.classes.name = {cats.name} ;
imdb.images.id = [] ;
imdb.sets = {'train', 'val', 'test'} ;

for c=1:numel(cats)
  ims = dir(fullfile(imdb.imageDir, imdb.classes.name{c}, '*.jpg'));
  %imdb.images.name{c} = fullfile(imdb.classes.name{c}, {ims.name}) ;
  imdb.images.name{c} = cellfun(@(S) fullfile(imdb.classes.name{c}, S), ...
    {ims.name}, 'Uniform', 0);
  imdb.images.label{c} = c * ones(1,numel(ims)) ;
  if numel(ims) ~= 100  
      numel(ims)
      cats(c)
      error('MIT indoor data inconsistent') ; 
  end
  %sets = [1 * ones(1,40), 2 * ones(1,40), 3 * ones(1,40)] ;
  %imdb.images.set{c} = sets(randperm(120)) ;
end
imdb.images.name = horzcat(imdb.images.name{:}) ;
imdb.images.label = horzcat(imdb.images.label{:}) ;
%imdb.images.set = horzcat(imdb.images.set{:}) ;
imdb.images.id = 1:numel(imdb.images.name) ;

% Note that there is no validation data in the original split
for s = [1 3]
  list = textread(fullfile(datasetDir, 'meta', ...
    sprintf('%s.txt', imdb.sets{s})),'%s') ;
  imdb.images.set(find(ismember(imdb.images.name, list))) = s ;
end

% hack
sel_train = find(imdb.images.set == 1);
imdb.images.set(sel_train(1 : 4 : end)) = 2;

imdb.segments = imdb.images ;
imdb.segments.imageId = imdb.images.id ;
% no segment masks

% make this compatible with the OS imdb
imdb.meta.classes = imdb.classes.name ;
imdb.meta.inUse = true(1,numel(imdb.meta.classes)) ;
imdb.segments.difficult = false(1, numel(imdb.segments.id)) ;

