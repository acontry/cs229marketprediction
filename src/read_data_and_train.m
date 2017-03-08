%% Read in data
Connect = yahoo;
securities = {'AAPL', 'AIG', 'ALL', 'AMZN', 'APA', 'APC', 'AXP', 'BA', ...
'BAC', 'BAX', 'BIIB', 'BK', 'BMY', 'BRK.B', 'C', 'CAT', 'CL', 'CMCSA', ...
'COF', 'COP', 'COST', 'CSCO', 'CVS', 'CVX', 'DD', 'DIS', 'DOW', 'MSFT'};

%securities = {'COST'};

for i = 1:length(securities)
    
    c = fetch(Connect, securities(i), 'Adj Close', '2001-01-01', '2014-11-01');
    v = fetch(Connect, securities(i), 'Volume', '2001-01-01', '2014-11-01');
    op = fetch(Connect, securities(i), 'Open', '2001-01-01', '2014-11-01');
    lo = fetch(Connect, securities(i), 'Low', '2001-01-01', '2014-11-01');
    hi = fetch(Connect, securities(i), 'High', '2001-01-01', '2014-11-01');
    
    if length(c) ~= 3480
        error('%s is the wrong length!',securities{i});
    end
    
    if i == 1
        open = op(:,2);
        high = hi(:,2);
        low = lo(:,2);
        close = c(:,2);
        date = c(:,1);
        volume = v(:,2);
    else
        open(:,end+1) = op(:,2);
        high(:,end+1) = hi(:,2);
        low(:,end+1) = lo(:,2);
        close(:,end+1) = c(:,2);
        volume(:,end+1) = v(:,2);
    end
end

open = flipud(open);
high = flipud(high);
low = flipud(low);
close = flipud(close);  % Make last entry the latest
volume = flipud(volume);

date = flipud(date);

clear c;
clear v;
clear op;
clear lo;
clear hi;

%% Volatility calculation

log_ret = log(close(2:end,:) ./ close(1:end-1,:));
volatility = sqrt(252) * std(log_ret);

%% Calculate daily gain and transform data for training
% In this section the data modifies itself so only run once after fetching data!
holdout_pct = 0.2;  % Hold out this data for testing
days_ahead = 1;  % Predict gain/loss this many days in advance

gain = (close(days_ahead + 1:end,:) - close(1:end-days_ahead,:)) ./ ...
    close(1:end-days_ahead,:);
gain_bool = gain > 0;

close = close(1:end-days_ahead,:);  % Knock off last entry of close to match gain
volume = volume(1:end-days_ahead,:);
open = open(1:end-days_ahead,:);
high = high(1:end-days_ahead,:);
low = low(1:end-days_ahead,:);


%% Save 2 years for testing
close_all = close;
close = close(1:end-504,:);

volume_all = volume;
volume = volume(1:end-504,:);

open_all = open;
open = open(1:end-504,:);

high_all = high;
high = high(1:end-504,:);

low_all = low;
low = low(1:end-504,:);

gain_bool_all = gain_bool;
gain_bool = gain_bool(1:end-504,:);


%% Set up Indicators
s = 1;  % Security index
avg10 = tsmovavg(close(:,s), 's', 10, 1);
avg30 = tsmovavg(close(:,s), 's', 30, 1);
avg100 = tsmovavg(close(:,s), 's', 100, 1);
avg300 = tsmovavg(close(:,s), 's', 300, 1);

oroc3 = prcroc(open(:,s), 3);
oroc10 = prcroc(open(:,s), 10);
oroc30 = prcroc(open(:,s), 30);
oroc100 = prcroc(open(:,s), 100);
oroc300 = prcroc(open(:,s), 300);

hroc3 = prcroc(high(:,s), 3);
hroc10 = prcroc(high(:,s), 10);
hroc30 = prcroc(high(:,s), 30);
hroc100 = prcroc(high(:,s), 100);
hroc300 = prcroc(high(:,s), 300);

lroc3 = prcroc(low(:,s), 3);
lroc10 = prcroc(low(:,s), 10);
lroc30 = prcroc(low(:,s), 30);
lroc100 = prcroc(low(:,s), 100);
lroc300 = prcroc(low(:,s), 300);

croc3 = prcroc(close(:,s), 3);
croc10 = prcroc(close(:,s), 10);
croc30 = prcroc(close(:,s), 30);
croc100 = prcroc(close(:,s), 100);
croc300 = prcroc(close(:,s), 300);
croc600 = prcroc(close(:,s), 600);

vroc3 = prcroc(volume(:,s), 3);
vroc10 = prcroc(volume(:,s), 10);
vroc30 = prcroc(volume(:,s), 30);
vroc100 = prcroc(volume(:,s), 100);
vroc300 = prcroc(volume(:,s), 300);

mom = tsmom(close(:,s), 5);

[macdvec, nineperma] = macd(close(:,s));


data = [croc3 croc10 croc30 croc100 croc300 vroc3 vroc10 vroc30 vroc100 vroc300 oroc10];
%data = [proc1 vroc1];


%% Try out training a SVM
tic
[train_pct_correct, pct_correct] = test_svm(data, gain_bool, 3, holdout_pct);
toc
fprintf('SVM train=%.3f test=%.3f\n',train_pct_correct,pct_correct);

%% Try out training logistic regression
[train_log_pct_correct, log_pct_correct] = test_log(data, gain_bool, 3, holdout_pct);


fprintf('LOG train=%.3f test=%.3f\n',train_log_pct_correct,log_pct_correct);

%% Try out forward search, both SVM and logistic regression
all_data = [oroc3 oroc10 oroc30 oroc100 oroc300 ...
    hroc3 hroc10 hroc30 hroc100 hroc300 ...
    lroc3 lroc10 lroc30 lroc100 lroc300 ...
    croc3 croc10 croc30 croc100 croc300 ...
    vroc3 vroc10 vroc30 vroc100 vroc300];

tic
[best_pct, best_f, best_f_idx] = forward_search(@test_svm, all_data, gain_bool, s, holdout_pct, 12);
toc
tic
[best_pct_log, best_f_log, best_f_idx_log] = forward_search(@test_log, all_data, gain_bool, s, holdout_pct, 12);
toc

%% The big training block!
% For each security, calculate indicators and perform forward search to
% find up to 12 indicators to maximize correct predictions.
best_pct_total = zeros(size(securities));
best_f_total = cell(size(securities));
best_f_total_idx = cell(size(securities));

best_pct_total_log = zeros(size(securities));
best_f_total_log = cell(size(securities));
best_f_total_idx_log = cell(size(securities));

tic
parfor s = 1:length(securities)
    s

    % Calc features
    oroc3 = prcroc(open(:,s), 3);
    oroc10 = prcroc(open(:,s), 10);
    oroc30 = prcroc(open(:,s), 30);
    oroc100 = prcroc(open(:,s), 100);
    oroc300 = prcroc(open(:,s), 300);

    hroc3 = prcroc(high(:,s), 3);
    hroc10 = prcroc(high(:,s), 10);
    hroc30 = prcroc(high(:,s), 30);
    hroc100 = prcroc(high(:,s), 100);
    hroc300 = prcroc(high(:,s), 300);

    lroc3 = prcroc(low(:,s), 3);
    lroc10 = prcroc(low(:,s), 10);
    lroc30 = prcroc(low(:,s), 30);
    lroc100 = prcroc(low(:,s), 100);
    lroc300 = prcroc(low(:,s), 300);

    croc3 = prcroc(close(:,s), 3);
    croc10 = prcroc(close(:,s), 10);
    croc30 = prcroc(close(:,s), 30);
    croc100 = prcroc(close(:,s), 100);
    croc300 = prcroc(close(:,s), 300);
    croc600 = prcroc(close(:,s), 600);

    vroc3 = prcroc(volume(:,s), 3);
    vroc10 = prcroc(volume(:,s), 10);
    vroc30 = prcroc(volume(:,s), 30);
    vroc100 = prcroc(volume(:,s), 100);
    vroc300 = prcroc(volume(:,s), 300);

    all_data = [oroc3 oroc10 oroc30 oroc100 oroc300 ...
        hroc3 hroc10 hroc30 hroc100 hroc300 ...
        lroc3 lroc10 lroc30 lroc100 lroc300 ...
        croc3 croc10 croc30 croc100 croc300 ...
        vroc3 vroc10 vroc30 vroc100 vroc300];


    [best_pct, best_f, best_f_idx] = forward_search(@test_svm, all_data, gain_bool, s, 0.2, 12);

    [~,idx] = max(best_pct);
    best_pct_total(s) = best_pct(idx);
    best_f_total{s} = best_f{idx};
    best_f_total_idx{s} = best_f_idx{idx};

    [best_pct, best_f, best_f_idx] = forward_search(@test_log, all_data, gain_bool, s, 0.2, 12);

    [~,idx] = max(best_pct);
    best_pct_total_log(s) = best_pct(idx);
    best_f_total_log{s} = best_f{idx};
    best_f_total_idx_log{s} = best_f_idx{idx};

end 
toc

%% Train SVM on best indicators and predict daily gain/loss for holdout data

for s = 1:28
    % Calc features
    oroc3 = prcroc(open_all(:,s), 3);
    oroc10 = prcroc(open_all(:,s), 10);
    oroc30 = prcroc(open_all(:,s), 30);
    oroc100 = prcroc(open_all(:,s), 100);
    oroc300 = prcroc(open_all(:,s), 300);

    hroc3 = prcroc(high_all(:,s), 3);
    hroc10 = prcroc(high_all(:,s), 10);
    hroc30 = prcroc(high_all(:,s), 30);
    hroc100 = prcroc(high_all(:,s), 100);
    hroc300 = prcroc(high_all(:,s), 300);

    lroc3 = prcroc(low_all(:,s), 3);
    lroc10 = prcroc(low_all(:,s), 10);
    lroc30 = prcroc(low_all(:,s), 30);
    lroc100 = prcroc(low_all(:,s), 100);
    lroc300 = prcroc(low_all(:,s), 300);

    croc3 = prcroc(close_all(:,s), 3);
    croc10 = prcroc(close_all(:,s), 10);
    croc30 = prcroc(close_all(:,s), 30);
    croc100 = prcroc(close_all(:,s), 100);
    croc300 = prcroc(close_all(:,s), 300);
    croc600 = prcroc(close_all(:,s), 600);

    vroc3 = prcroc(volume_all(:,s), 3);
    vroc10 = prcroc(volume_all(:,s), 10);
    vroc30 = prcroc(volume_all(:,s), 30);
    vroc100 = prcroc(volume_all(:,s), 100);
    vroc300 = prcroc(volume_all(:,s), 300);

    all_data = [oroc3 oroc10 oroc30 oroc100 oroc300 ...
        hroc3 hroc10 hroc30 hroc100 hroc300 ...
        lroc3 lroc10 lroc30 lroc100 lroc300 ...
        croc3 croc10 croc30 croc100 croc300 ...
        vroc3 vroc10 vroc30 vroc100 vroc300];
    
    % Grab test data
    %end_data = all_data(end-503:end,:);
    %all_data = all_data(1:end-504,:);
    
    idx = best_f_total_idx{s};
    s_data = [];
    %idx = my_idx;
    for i = 1:length(idx)
        s_data = [s_data all_data(:,idx(i))];
    end
    
    series_data = s_data(end-503:end,:);
    train_data = s_data(1:end-504,:);
    
    
    
   boxconstraint = 1;
   rbf_sigma = 1;
   svmStruct = svmtrain(train_data, gain_bool(:,s), ...
    'Kernel_Function', 'rbf', 'boxconstraint', boxconstraint, ...
    'rbf_sigma', rbf_sigma);

pred_gain_bool(:,s) = svmclassify(svmStruct, series_data);
    
end

%% Simple trading algorithm simulation for a single security. If the 
% closing price tomorrow is expected to be higher (lower) then buy (sell).
s = 1;
series_close = close_all(end-503:end,s);

equity = 10000;
stock = 0;
value = zeros(size(series_close));
for i = 1:length(series_close)
    if pred_gain_bool(i,s) == 1;
       if equity > 0
          stock = stock + equity / series_close(i); 
          equity = 0;
       end
    else
        if stock > 0
           equity = stock * series_close(i);
           stock = 0;
        end
    end
    value(i) = equity + stock * series_close(i);
end