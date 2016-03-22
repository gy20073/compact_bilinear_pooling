% modify the CUB imdb structure such that only the first 100 classes are
% included
imdb=load(consts('CUB','imdb'));
imdb.segments=[];
retain=(imdb.images.label < 101);
fields=fieldnames(imdb.images);
imdb.images
%%
for i=1:numel(fields)
    name=fields{i};
    imdb.images.(name)=imdb.images.(name)(retain);
end
imdb.images
%%
save(consts('CUB', 'imdb'), '-struct', 'imdb');