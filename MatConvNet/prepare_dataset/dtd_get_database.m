function imdb = dtd_get_database(dtdDir, varargin)
opts.seed = 1 ;
opts.joint = 0;
opts = vl_argparse(opts, varargin) ;

rng(opts.seed) ;

imdb.imageDir = fullfile(dtdDir, 'images') ;
imdb.maskDir = fullfile(dtdDir, 'mask') ;

cats = dir(imdb.imageDir) ;
cats = cats([cats.isdir] & ~ismember({cats.name}, {'.','..'})) ;
imdb.classes.name = {cats.name} ;
imdb.images.id = [] ;
imdb.sets = {'train', 'val', 'test'} ;

for c=1:numel(cats)
  ims = dir(fullfile(imdb.imageDir, imdb.classes.name{c}, '*.jpg'));
  imdb.images.name{c} = cellfun(@(S) fullfile(imdb.classes.name{c}, S), ...
    {ims.name}, 'Uniform', 0);
  imdb.images.label{c} = c * ones(1,numel(ims)) ;
  if numel(ims) ~= 120, error('DTD folder appears inconsistent') ; end
end
imdb.images.name = horzcat(imdb.images.name{:}) ;
imdb.images.label = horzcat(imdb.images.label{:}) ;
imdb.images.id = 1:numel(imdb.images.name) ;

for s = 1:3
  list = textread(fullfile(dtdDir, 'labels', ...
    sprintf('%s%d.txt', imdb.sets{s}, opts.seed)),'%s') ;
  ok = ismember(imdb.images.name, list) ;
  imdb.images.set(find(ok)) = s ;
end
if any(~ismember(imdb.images.set, 1:3)), error('DTD appears inconsistent') ; end

imdb.segments = imdb.images ;
imdb.segments.imageId = imdb.images.id ;
imdb.segments.mask = strrep(imdb.images.name, 'image', 'mask') ;

imdb.meta.classes = imdb.classes.name ;
imdb.meta.inUse = true(1, numel(imdb.meta.classes)) ;
imdb.segments.difficult = false(1, numel(imdb.segments.id)) ;

if (opts.joint)
    annoPath = fullfile(dtdDir, 'labels', 'labels_joint_anno.txt');
    imdb.images.label = -1 * ones(numel(imdb.meta.classes), numel(imdb.images.name));
    fid = fopen(annoPath);
    if (fid > 0)
        lines = textscan(fid, '%s', 'Delimiter', '\n');
        lines = lines{1};
        for ii = 1 : numel(lines)
            parts = strsplit(lines{ii}, ' ');
            parts = parts(1: end - 1);
            imName = parts{1};
            parts = parts{2 : end};
            keyAttr = imName(1 : find(imName == '/') - 1);
            imId = strcmp(imdb.images.name, imName);
            imdb.images.label(ismember(imdb.meta.classes, parts), imId) = 1;
            imdb.images.label(ismember(imdb.meta.classes, keyAttr), imId) = 1;
        end
    end
    imdb.segments.label = imdb.images.label;
end




