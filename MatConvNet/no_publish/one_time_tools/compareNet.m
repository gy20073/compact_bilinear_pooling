function out=compareNet(net1, net2)
    out=[];
    for il=1:numel(net1.layers)
        if isfield(net1.layers{il}, 'weights')
            w1=net1.layers{il}.weights;
            w2=net2.layers{il}.weights;
            for i=1:2
                v=mean(abs(w1{i}(:)));
                dif=mean(abs(w1{i}(:)-w2{i}(:)));
                fprintf('Layer %d, weight %d, mean1 %f, mean diff %f, relative diff %f\n',...
                    il, i, v, dif, dif/(v+1e-10));
                if isempty(out)
                    out=dif/(v+1e-10);
                end
            end
        end
    end
       
end