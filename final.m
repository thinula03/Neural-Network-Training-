% Define users and categories for time domain
num_users = 10;
categories = {'Acc_TimeD_FDay', 'Acc_TimeD_MDay'};

% Base path for data files
base_path = '/Users/thinuladamsith/Documents/MATLAB/data/';

% Initialize variables for training and testing data
trainFeatures = [];
trainLabels = [];
testFeatures = [];
testLabels = [];

% Loop through users to load FDay (training) and MDay (testing) data
for user_id = 1:num_users
    % Load FDay (Training Data)
    fday_file = sprintf('U%02d_%s.mat', user_id, categories{1});
    fday_path = fullfile(base_path, fday_file);
    if isfile(fday_path)
        data = load(fday_path);
        field_name = fieldnames(data);
        fday_features = data.(field_name{1});
        trainFeatures = [trainFeatures; fday_features]; %#ok<AGROW>
        trainLabels = [trainLabels; repelem(user_id, size(fday_features, 1))']; %#ok<AGROW>
    else
        warning('File not found: %s', fday_path);
    end

    % Load MDay (Testing Data)
    mday_file = sprintf('U%02d_%s.mat', user_id, categories{2});
    mday_path = fullfile(base_path, mday_file);
    if isfile(mday_path)
        data = load(mday_path);
        field_name = fieldnames(data);
        mday_features = data.(field_name{1});
        testFeatures = [testFeatures; mday_features]; %#ok<AGROW>
        testLabels = [testLabels; repelem(user_id, size(mday_features, 1))']; %#ok<AGROW>
    else
        warning('File not found: %s', mday_path);
    end
end

% Display dimensions of training and testing datasets
disp(['Training Data Dimensions: ', num2str(size(trainFeatures))]);
disp(['Testing Data Dimensions: ', num2str(size(testFeatures))]);

% Normalize the features (Z-score normalization)
[trainFeatures, mu, sigma] = zscore(trainFeatures);
testFeatures = (testFeatures - mu) ./ sigma;

% One-hot encode the labels
num_classes = num_users; % Each user is a class
trainLabelsEncoded = full(ind2vec(trainLabels')); % Convert to one-hot encoded matrix
testLabelsEncoded = full(ind2vec(testLabels'));   % Convert to one-hot encoded matrix

% Define the feedforward neural network
hidden_layers = [128, 64, 32]; % Define the architecture
net = patternnet(hidden_layers);

% Set training parameters
net.trainParam.epochs = 500; % Maximum number of epochs
net.trainParam.lr = 0.01;    % Learning rate
net.trainParam.goal = 1e-6;  % Performance goal
net.trainParam.showWindow = true; % Display training progress

% Train the neural network
[net, tr] = train(net, trainFeatures', trainLabelsEncoded);

% Predict the test data
testPredictions = net(testFeatures');
[~, predictedClasses] = max(testPredictions, [], 1); % Get predicted class indices

% Evaluate the performance
accuracy = sum(predictedClasses' == testLabels) / length(testLabels) * 100;
disp(['Test Accuracy: ', num2str(accuracy), '%']);

% Confusion Matrix
confMatrix = confusionmat(testLabels, predictedClasses');
figure;
confusionchart(confMatrix);
title('Confusion Matrix for Test Data');

% Display Precision, Recall, and F1-Score
true_positive = diag(confMatrix);
precision = true_positive ./ sum(confMatrix, 1)';
recall = true_positive ./ sum(confMatrix, 2);
f1_score = 2 * (precision .* recall) ./ (precision + recall);

% Handle cases where precision or recall is NaN
precision(isnan(precision)) = 0;
recall(isnan(recall)) = 0;
f1_score(isnan(f1_score)) = 0;

disp(['Precision: ', num2str(mean(precision) * 100), '%']);
disp(['Recall: ', num2str(mean(recall) * 100), '%']);
disp(['F1-Score: ', num2str(mean(f1_score) * 100), '%']);

% Plot training performance
figure;
plotperform(tr);
title('Training Performance');

% Feature Importance via PCA
[coeff, ~, ~, ~, explained] = pca(trainFeatures);
cumulativeExplained = cumsum(explained);

% Bar plot for first principal component
figure;
bar(coeff(:, 1));
title('Feature Importance (PCA - First Principal Component)');
xlabel('Feature Index');
ylabel('Contribution to Principal Component');

% Cumulative variance explained by principal components
figure;
plot(cumulativeExplained, '-o');
title('Cumulative Variance Explained by Principal Components');
xlabel('Number of Components');
ylabel('Variance Explained (%)');
grid on;

% Boxplot of feature distributions
num_features_to_plot = min(5, size(trainFeatures, 2)); % Limit to 5 features
figure;
for feature_idx = 1:num_features_to_plot
    subplot(num_features_to_plot, 1, feature_idx);
    boxplot(trainFeatures(:, feature_idx), trainLabels, 'Colors', 'b', 'Symbol', 'o');
    title(['Boxplot for Feature ', num2str(feature_idx)]);
    xlabel('User ID');
    ylabel('Feature Value');
    grid on;
end
sgtitle('Boxplots of Selected Features Across Users');

% Pairwise scatter plots
figure;
num_features_to_plot = 5;
plot_idx = 1;
for i = 1:num_features_to_plot
    for j = i+1:num_features_to_plot
        subplot(num_features_to_plot-1, num_features_to_plot-1, plot_idx);
        scatter(trainFeatures(:, i), trainFeatures(:, j), 30, trainLabels, 'filled');
        xlabel(['Feature ', num2str(i)]);
        ylabel(['Feature ', num2str(j)]);
        grid on;
        plot_idx = plot_idx + 1;
    end
end
sgtitle('Pairwise Scatter Plots of Selected Features');

% Training Progress Metrics
figure;
plot(tr.epoch, tr.perf, '-o', 'DisplayName', 'Training Loss');
hold on;
plot(tr.epoch, tr.vperf, '-x', 'DisplayName', 'Validation Loss');
title('Training and Validation Loss Over Epochs');
xlabel('Epoch');
ylabel('Loss');
legend;
grid on;

% Heatmap of feature variance
figure;
heatmap(var(trainFeatures, 0, 1), 'Colormap', parula, ...
    'XLabel', 'Feature Index', 'Title', 'Feature Variance Heatmap');

% Save Results
save('trained_network.mat', 'net');
save('confusion_matrix.mat', 'confMatrix');
save('performance_metrics.mat', 'accuracy', 'precision', 'recall', 'f1_score');
disp('All results saved successfully!');

