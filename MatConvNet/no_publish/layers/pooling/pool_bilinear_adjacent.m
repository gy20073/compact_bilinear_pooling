function [trainFs, testFs]=pool_bilinear_adjacent(trainFs, testFs)
    trainFs=one(trainFs);
    testFs=one(testFs);
end

function out=loc(h, w, i, j)
    out=(j-1)*h+i;
end

function out=one(fs)
   [h, w, c, n]=size(fs);
   fs=reshape(fs, h*w, c, n);

   fs=permute(fs, [2, 1, 3]);
   out=zeros(c*c, n, 'single');
   parfor i=1:n
       cur_image=fs(:, :, i);
       cur_image=sparse(double(cur_image));
       if mod(i,100)==1
           fprintf('i=%d\n',i);
       end
       temp=sparse(c,c);
       for j=1:(h-1)
           for k=1:(w-1)
               v=cur_image(:, loc(h, w, j, k));
               vr=cur_image(:,loc(h, w, j+1, k));
               vd=cur_image(:,loc(h, w, j, k+1));
               temp=temp + v*(vr+vd)';
           end
       end
       temp=full(temp);
       out(:, i)=temp(:)/(h*w);
   end
end
