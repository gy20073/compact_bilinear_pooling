function [dist_standard, dist_appx]=test_chi2_sub(t1, t2, p1, p2, groundTruthDist)
    if nargin<=4 
        groundTruthDist=@dist_chi2;
    end
    dist_standard=groundTruthDist(t1, t2);
    if nargin<=2
        proj=@(fs)proj_chi2(fs, 5, 0.3);
        p1=proj(t1);
        p2=proj(t2);
    end
    dist_appx=dot(p1, p2, 1);

    err=abs(dist_standard - dist_appx);
    sprintf('max absolute error %f \n',max(err));
    figure
    subplot(1,2,1);
    hist(err,100);
    title('Distribution of absolute error')

    rel_err=err./(dist_standard + 1e-10);
    sprintf('max relative error %f \n',max(rel_err));
    subplot(1,2,2);
    hist(rel_err,100);
    title('Distribution of relative error')
end
