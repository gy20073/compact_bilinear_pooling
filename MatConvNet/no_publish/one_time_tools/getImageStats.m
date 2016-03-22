% compute average image, rgb mean and covariance. 
% -------------------------------------------------------------------------
function [averageImage, rgbMean, rgbCovariance] = getImageStats(imdb, opts)
% -------------------------------------------------------------------------
    train = find(imdb.images.set == 1) ;
    train = train(1: 101: end);
    bs = 256 ;
    fn = getBatchWrapper(opts) ;
    for t=1:bs:numel(train)
      batch_time = tic ;
      batch = train(t:min(t+bs-1, numel(train))) ;
      fprintf('collecting image stats: batch starting with image %d ...', batch(1)) ;
      temp = fn(imdb, batch) ;
      z = reshape(permute(temp,[3 1 2 4]),3,[]) ;
      n = size(z,2) ;
      avg{t} = mean(temp, 4) ;
      rgbm1{t} = sum(z,2)/n ;
      rgbm2{t} = z*z'/n ;
      batch_time = toc(batch_time) ;
      fprintf(' %.2f s (%.1f images/s)\n', batch_time, numel(batch)/ batch_time) ;
    end
    averageImage = mean(cat(4,avg{:}),4) ;
    rgbm1 = mean(cat(2,rgbm1{:}),2) ;
    rgbm2 = mean(cat(3,rgbm2{:}),3) ;
    rgbMean = rgbm1 ;
    rgbCovariance = rgbm2 - rgbm1*rgbm1' ;
end

