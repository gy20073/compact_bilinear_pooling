% return a get batch function
% -------------------------------------------------------------------------
function fn = getBatchWrapper(opts)
% -------------------------------------------------------------------------
    fn = @(imdb,batch) getBatch(imdb,batch,opts) ;
end

% -------------------------------------------------------------------------
function [im,labels] = getBatch(imdb, batch, opts)
% -------------------------------------------------------------------------
    images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
    im = cnn_imagenet_get_batch(images, opts, ...
                                'prefetch', nargout == 0) ;
    labels = imdb.images.label(batch) ;
end
