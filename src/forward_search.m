function [best_pct, best_f, best_f_idx] = forward_search(f_handle, data, gain_bool, s, holdout_pct, N)
% Performs forward search on a dataset to find most relevant indicators.
%
% f_handle = function handle to training function which follows the
% signature:
% [train_pct_correct, pct_correct] = f(data, gain_bool, s, holdout_pct)
%
% data = Dataset to train and test on. Rows are timseries entries and
% columns are indicators (or features)
%
% gain_bool = Boolean indicator for whether a particular timeseries entry
% will gain in value the following day (the result we're trying to
% predict).
%
% s = index of security in the dataset to work on
%
% holdout_pct = Fraction of data to hold out from training for use in
% testing.
%
% N = Number of features to select using forward selection
if nargin < 6
    N = size(data,2);
end

best_pct = zeros(N,1);
best_f = cell(N,1);
best_f_idx = cell(N,1);

trial_features = [];
trial_idx = [];
% Add one feature at a time
for n = 1:N
    % Try each feature and save the one which results in highest prediction
    % rate
    fprintf('Fwd search n = %d\n',n);
    for i = 1:size(data,2)
        % If we haven't already chosen this feature 
        if isempty(find(trial_idx == i,1))
            f = [trial_features data(:,i)];
            f_idx = [trial_idx i];
            try
                [~,pct] = f_handle(f, gain_bool, s, holdout_pct);
            catch
                pct = 0;
                fprintf('No convergence for %d\n',i);
            end
            
            if pct > best_pct(n)
                best_pct(n) = pct;
                best_f{n} = f;
                best_f_idx{n} = f_idx;
            end
            
        end
    end
    
    trial_features = best_f{n};
    trial_idx = best_f_idx{n};
end