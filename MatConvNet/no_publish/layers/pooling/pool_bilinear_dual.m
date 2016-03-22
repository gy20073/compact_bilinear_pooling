% this one is pretty bad, just for exploration, abandoned
function [trainFs, testFs]=pool_bilinear_dual(trainFs, testFs)
    trainFs=one(trainFs);
    testFs=one(testFs);
end

function out=one(fs)
   [h, w, c, n]=size(fs);
   hw=h*w;
   fs=reshape(fs, hw, c,n);
   
   out=zeros(hw*hw, n, 'single');
   parfor i=1:n
       cur_image=fs(:, :, i);
       cur_image=sparse(double(cur_image));
       if mod(i,100)==1
           fprintf('i=%d\n',i);
       end
       temp=sparse(hw,hw);
       for j=1:c
           v=cur_image(:, j);
           temp=temp + v*v';
       end
       temp=full(temp);
       out(:, i)=temp(:)/(h*w);
   end
end
