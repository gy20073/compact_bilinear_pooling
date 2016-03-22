function [trainFs, testFs]=dim_reduction_sum(trainFs, testFs, ratio)
    fun=@jump;

    trainFs=fun(trainFs, ratio);
    testFs=fun(testFs, ratio);
end

function out=continous(fs, ratio)
    [dim, n]=size(fs);
    ansDim=ceil(dim/ratio);
    out=zeros(ansDim, n, 'single');
    % continuous mix up
    for i=1:floor(dim/ratio)
        inter=((i-1)*ratio+1): (i*ratio);
        out(i, :)=mean(fs(inter, :));
    end
    if mod(dim, ratio)~=0
        out(ansDim, :)=mean(fs((ansDim-1)*ratio+1 : end, :));
    end
end

function out=jump(fs, ratio)
    [dim, n]=size(fs);
    ansDim=floor(dim/ratio);
    out=zeros(ansDim, n, 'single');
    nend=ansDim*ratio;

    for i=1:ratio
        inter=i:ratio:nend;
        out=out+fs(inter,:);
    end
    out=out/ratio;
end
