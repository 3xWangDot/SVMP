% Example code for CVPR18 paper: 
% Video Representation Learning Using Discriminative Pooling. 
% (https://arxiv.org/pdf/1803.10628.pdf)
% Author: Jue Wang; Anoop Cherian; Fatih Porikli; Stephen Gould;
function demo_SVMP
addpath(genpath('liblinear-2.1'))
run('liblinear-2.1/matlab/make.m') % prepare liblinear
load('HMDB-51_s1_train_sub_10class.mat');
load('HMDB-51_s1_test_sub_10class.mat');
data_tr=[];
data_va=[];
num_tr=length(train_data);
num_va=length(test_data);
for i=1:1:num_tr
    fprintf('Current processing Train Data %d of %d\n',i,num_tr);
    positive=normr(train_data{i}); %norm both positive and negative
    s=size(positive);
    negative=normr(rand(s(1),s(2)));%negative bag, random number is the default here
    data_tr(i,:)=SVMP(positive,negative);
end
for i=1:1:num_va
    fprintf('Current processing Test Data %d of %d\n',i,num_va);
    positive=normr(test_data{i});
    s=size(positive);
    negative=normr(rand(s(1),s(2)));
    data_va(i,:)=SVMP(positive,negative);
end
data_va=sparse(double(data_va));
test_label=test_label';
data_tr=sparse(double(data_tr));
train_label=train_label';
svmmodel=train(train_label,data_tr,['-s 1 -c 0.01 -q']);
[predict_labe, accu, prob] = predict(test_label, data_va, svmmodel);
end