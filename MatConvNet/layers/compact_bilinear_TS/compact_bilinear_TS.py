# before running this script please install the python matlab engine:
# cd "matlabroot/extern/engines/python"
# Replace matlabroot should be like: cd "/usr/local/MATLAB/R2016a/extern/engines/python"
# sudo python setup.py install

# the installation could be in "/usr/bin/python", check "which python"
import matlab.engine
import numpy as np

eng=matlab.engine.start_matlab()

eng.run("../../setup_yang.m", nargout=0)

# the number of output channels and input channels
num_output= 3;
num_input = 2;
TS_layer=eng.yang_compact_bilinear_TS_nopool_2stream(
	matlab.double([num_output]), 
	matlab.double([num_input, num_input]),
	1.0);

def np2mat(x):
	y=x.tolist();
	z=matlab.single(y);
	out=eng.permute(z, matlab.double([4,3,2,1]));
	return out;

def mat2np(x):
	y=eng.permute(x, matlab.double([4,3,2,1]));
	out=np.array(y);	
	return out;

def compact_TS_forward(x):
	# x is a numpy array of size N*C*H*W
	x=np2mat(x);
	y=eng.forward(TS_layer, [x, x], [])
	out=mat2np(y[0]);
	return out;
	# output is size: N*num_output*H*W

def test_forward():
	x=np.reshape(range(120), [3,2,4,5]);
	y=compact_TS_forward(x);
	return y;

def compact_TS_backward(x, dzdy):
	x=np2mat(x);
	dzdy=np2mat(dzdy);
	t=eng.backward(TS_layer, [x, x], [], [dzdy], nargout=2);
	print t
	derInputs=t[0];
	derParams=t[1]; # not used yet
	out=mat2np(derInputs[0]) + mat2np(derInputs[1]);
	return out;
	# output the same size of input x;	

def test_backward():
	x=np.reshape(range(120), [3,2,4,5]);
	dzdy=np.reshape(range(180), [3,3,4,5]);
	z=compact_TS_backward(x, dzdy);
	return z;
