function pre=yang_l2norm_backward(layer, pre, now)
	dzdy=now.dzdx;
	assert( (size(dzdy, 1)==1) && (size(dzdy, 2)==1) );
	dzdy=squeeze(dzdy);

	X=squeeze(pre.x);
	lambda=1./(sqrt(sum(X.^2, 1)) + 1e-10);

	dzdx=bsxfun(@times, lambda, dzdy) - bsxfun(@times, X, (lambda.^3) .* sum(X.*dzdy, 1));

	pre.dzdx=reshape(dzdx, 1, 1, size(dzdx, 1), size(dzdx, 2));
end
