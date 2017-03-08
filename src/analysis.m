%% Load stock data and training results. These two files are the main 
% results of read_data_and_train.m but other blocks in the file may need to
% be run to set up workspace to generate all plots below.
load results_all
load stocks

%% Calculate number of features used for each security

num_features = zeros(28,1);
num_features_log = zeros(28,1);
for i = 1:28
    num_features(i) = size(best_f_total{i},2);
    num_features_log(i) = size(best_f_total{i},2);
end


%% Stock volatility histogram
figure;
hist(volatility, 15)

set(gca,'linewidth',4)
set(gca,'fontsize',16)
xlabel('Annualized Volatility')
ylabel('Number of Stocks')

%% Forward search
figure;
hold on
plot(best_pct,'o','markerfacecolor','b')
plot(best_pct_log,'or','markerfacecolor','r')
box on
xlim([0 13])
ylim([0.50 0.56])
set(gca,'xtick',1:2:12)

set(gca,'linewidth',4)
set(gca,'fontsize',16)
xlabel('Number of Features')
ylabel('Percent Correct')

%% All stocks results summary
figure;
hold on
plot(best_pct_total,'o','markerfacecolor','b')
plot(best_pct_total_log,'or','markerfacecolor','r')
%legend('SVM','Logistic regression')
box on
set(gca,'linewidth',4)
set(gca,'fontsize',16)
xlabel('Stock')
ylabel('Percent Correct')

%% All stocks results vs. number of features
figure;
plot(num_features,best_pct_total,'o','markerfacecolor','b')
%set(gca,'xtick',[1 2 3 4 5 6])
xlim([2 13])

box on
set(gca,'linewidth',4)
set(gca,'fontsize',16)
xlabel('Number of Features')
ylabel('Percent Correct')

%% Results vs. volatility
figure;
plot(volatility,best_pct_total,'o','markerfacecolor','b')

box on
set(gca,'linewidth',4)
set(gca,'fontsize',16)
xlabel('Annualized Volatilty')
ylabel('Percent Correct')
%% Plot results of trading simulation vs. buy and hold benchmark
figure;
plot(series_close / series_close(1),'r','linewidth',2)
hold on
plot(value/10000,'linewidth',2)
xlim([0 500])
set(gca,'xtick',[0 250 500])
%set(gca,'ytick',[])

box on
set(gca,'linewidth',4)
set(gca,'fontsize',16)
xlabel('Day')
ylabel('Value')


%% All stocks results and what features were used
features = zeros(28,25);

for i = 1:28
   f = best_f_total_idx{i};
   for j = 1:length(f)
      features(i,f(j)) = 1; 
   end
end

figure;
imagesc(features)
