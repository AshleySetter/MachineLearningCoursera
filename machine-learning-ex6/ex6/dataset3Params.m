function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

best_f1score = 0;
best_accuracy = 0;

for i = 1:numel(C_vals)
    for j = 1:numel(sigma_vals)
        C = C_vals(i);
        sigma = sigma_vals(j);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        pred = svmPredict(model, Xval);
        accuracy = sum(yval==pred);
        num_true_pos = sum(yval==1 & yval==pred);
        num_pred_pos = sum(pred);
        num_actual_pos = sum(yval);
        precision = num_true_pos/num_pred_pos;
        recall = num_true_pos/num_actual_pos;
        f1score = 2*precision*recall/(precision+recall);
        if f1score > best_f1score
        %if accuracy > best_accuracy;
            best_accuracy = accuracy;
            best_f1score = f1score;
            best_C = C;
            best_sigma = sigma;
        end
    end
end

C = best_C
sigma = best_sigma

% =========================================================================

end
