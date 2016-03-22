function [trainFV, testFV]=normalize_matrix_log(trainFV, testFV, useLog, useSqrt)
    trainFV=one(trainFV, useLog, useSqrt);
    testFV=one(testFV, useLog, useSqrt);
end

function fs=one(fs, useLog, useSqrt)
    [c2, n]=size(fs);
    c=floor(sqrt(c2));
    fs=reshape(fs, c,c,n);
    if useLog
        parfor i=1:n
            fs(:,:,i)=logm(fs(:,:,i));
        end
    end
    if useSqrt
        parfor i=1:n
            fs(:,:,i)=sqrtm(fs(:,:,i));
        end
    end
    fs=reshape(fs, c2, n);
end
