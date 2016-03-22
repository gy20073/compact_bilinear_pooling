function out=getInter(iseg, segLen, upper)
% shorthand for get an interval with batchsize=segLen, 
% segment id=iseg (1-based) and upper bound=upper

    out=gpuArray( ((iseg-1)*segLen+1) : min(upper, iseg*segLen) );
end