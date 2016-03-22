function now=yang_l2norm_forward(layer, pre, now)
	X=pre.x;
	assert( (size(X, 1)==1) && (size(X, 2)==1) );
	X=squeeze(X);
	l2norm=@(x)bsxfun(@rdivide, x, sqrt(sum(x.^2, 1)));
	X=l2norm(X);
	X=reshape(X, 1, 1, size(X, 1), size(X, 2));
	now.x=X;
end
