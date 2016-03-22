function [trainFs, testFs]=pool_bilinear(trainFs, testFs, useSparse)
    trainFs=one(trainFs, useSparse);
    testFs=one(testFs, useSparse);

	% the second method is not faster than the first naive one, so abandoned. 
    %trainFs=two(trainFs);
    %testFs=two(testFs);
end

function out=one(fs, useSparse)
   [h, w, c, n]=size(fs);
   fs=reshape(fs, h*w, c,n);
   
   fs=permute(fs, [2,1,3]);
   out=zeros(c*c, n, 'single');
   parfor i=1:n
       cur_image=fs(:, :, i);
       if useSparse
           cur_image=sparse(double(cur_image));
       end 
       if mod(i,100)==1
           fprintf('i=%d\n',i);
       end
       if useSparse
           temp=sparse(c,c);
       else
           temp=zeros(c,c, 'single');
       end
       for j=1:w*h
           v=cur_image(:, j);
           temp=temp + v*v';
       end
       if useSparse
           temp=full(temp);
       end
       out(:, i)=temp(:)/(h*w);
   end
end


function out=two(fs)
   fs=permute(fs, [3,1,2,4]);   
   [c, h, w, n]=size(fs);
   fs=reshape(fs, c, h*w, n);
   
   out=zeros(c*c, n, 'single');
   tic
   for i=1:n
       if mod(i,100)==1
           fprintf('i=%d\n',i);
           toc
           tic
       end
        im=fs(:,:,i);
        A=repmat(im, c, 1);
        B=kron(ones(c,1), im);
        out(:, i)=dot(A,B, 2);
   end
end
