function [train_pct_correct, pct_correct] = test_log(data, gain_bool, s, holdout_pct)
% Train and test logistic regression on data

idx = round((1 - holdout_pct) * length(data));

train_data = data(1:idx,:);
test_data = data(idx+1:end,:);

train_gain_bool = gain_bool(1:idx,s);
test_gain_bool = gain_bool(idx+1:end,s);

B = mnrfit(train_data, train_gain_bool+1);
log_pred = mnrval(B,test_data);
log_pred_gain_bool = log_pred(:,2) > log_pred(:,1);

train_log_pred = mnrval(B,train_data(300:end,:));
train_log_correct_pred_bool = train_log_pred(:,2) > train_log_pred(:,1);

% Test and holdout % correct
log_correct_pred = log_pred_gain_bool == test_gain_bool;

train_log_correct_pred = train_log_correct_pred_bool == train_gain_bool(300:end);

train_pct_correct = sum(train_log_correct_pred) / length(train_log_correct_pred);
pct_correct = sum(log_correct_pred) / length(log_correct_pred);
