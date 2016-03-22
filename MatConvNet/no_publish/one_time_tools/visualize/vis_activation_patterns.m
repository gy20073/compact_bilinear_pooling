% explore the activation patterns
% assume trainFV is the MIT 13 + relu activations

act=trainFV;

[h,w,c,n]=size(act);
act=permute(act, [3,1,2,4]);
act=reshape(act, c, h*w*n);

select=1:16;
act=act(select, :);

map=hash_activation(act, 0);