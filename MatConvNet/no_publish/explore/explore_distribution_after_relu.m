%%
t=trainFV;
t=t(:);
a=t(t>0);
b=a(1:100:end);
clear t
%%
close all
histogram(b, 1000, 'Normalization', 'pdf');
%%
b=a(a>0);
b=b(1:100:end);
hold on

pd=fitdist(b, 'Exponential');
x_values=0.1:0.1:60;
y = pdf(pd,x_values);
plot(x_values,y,'LineWidth',2)
%% test my calculation of var(sqrt(x))
mu=mean(b);
v=sqrt(var(b));

vsb=var(sqrt(b)); % my calculation correct!
%% test whether the distribution is an exponential
c=exprnd(8, 1, 1000);

cutoff=10;
d=b(b>cutoff)-cutoff;
[h,p,kstat,critval] = lillietest(d, 'Distr','exp')

%% test whether is Gaussian
figure
t=trainFV(:);
b=t(1:100:end);
clear t
%%
close all
histogram(b, 1000, 'Normalization', 'pdf');
%%
hold on

pd=fitdist(b, 'Normal');
x_values=-60:0.1:60;
y = pdf(pd,x_values);
plot(x_values,y,'LineWidth',2)

%% explore the weights in the net
net=load(consts('MIT', 'vgg_m'));
%%
vl_simplenn_display(net)
% inspect the parameters in the last convolutional layer
para=net.layers{13};
w=para.weights{1};
b=para.weights{2};
%%
size(w)
t=w(:,:,:,:);
t=t(1:10:end);
histogram(t, 1000);
% no special trend in the weights

%% explore image original histogram
close all
a=imread('peppers.png');
imshow(a);
for i=1:3
    t=a(:,:,i);
    t=t(:);
    figure
    hist(double(t), 60);
end


