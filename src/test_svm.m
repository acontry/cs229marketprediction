function [train_pct_correct, pct_correct] = test_svm(data, gain_bool, s, holdout_pct)
% Train and test SVM on data

idx = round((1 - holdout_pct) * length(data));

train_data = data(1:idx,:);
test_data = data(idx+1:end,:);

train_gain_bool = gain_bool(1:idx,s);
test_gain_bool = gain_bool(idx+1:end,s);

boxconstraint = 1;
rbf_sigma = 1;

svmStruct = svmtrain(train_data, train_gain_bool, ...
    'Kernel_Function', 'rbf', 'boxconstraint', boxconstraint, ...
    'rbf_sigma', rbf_sigma);

pred_gain_bool = svmclassify(svmStruct, test_data);

% Test and holdout correct % calc
correct_pred = pred_gain_bool == test_gain_bool;
train_correct_pred = svmclassify(svmStruct, train_data(300:end,:)) == train_gain_bool(300:end);

train_pct_correct = sum(train_correct_pred) / length(train_correct_pred);
pct_correct = sum(correct_pred) / length(correct_pred);
