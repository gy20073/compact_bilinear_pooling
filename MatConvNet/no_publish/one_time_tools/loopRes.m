function loopRes(res, fun)
    for i=1:numel(res)
        ires=res(i);
        eval([fun '(ires, i);']);
    end
end

function out=hasOneNan(x)
    out=any(isnan(x(:)));
end

function hasNan(res, ilayer) 
    out=0;
    hasField=0;
    if isfield(res, 'x')
        hasField=1;
        out=out || hasOneNan(res.x);
    end
    
    if isfield(res, 'dzdx')
        hasField=1;
        out=out || hasOneNan(res.dzdx);
    end
    if isfield(res, 'dzdw') && ~isempty(res.dzdw)
        hasField=1;
        out=out || hasOneNan(res.dzdw{1}) || hasOneNan(res.dzdw{2});
    end
    if hasField
        fprintf('Layer %d has nan=%d\n', ilayer, out);
    end
end

