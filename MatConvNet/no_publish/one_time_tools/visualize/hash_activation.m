function out=hash_activation(in, thres)
    x=int64(in>thres);
    prime=945633441251;
    % assume x is 2 dimensional, (c, n*h*w)
    [c, n]=size(x);
    hashvs=zeros(1, n, 'int64');
    fprintf('constructing hash values\n');
    for i=1:c
        hashvs=mod(hashvs*2+x(i, :), prime);
    end
    fprintf('constructing hash maps\n');
    out=containers.Map(num2cell(hashvs), mat2cell(in, c, ones(1, n)));
    fprintf('done\n')
end
