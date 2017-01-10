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


allC = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
allSigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
allC = [0.3, 1, 3];
allSigma = [0.03, 0.1, 0.3];
minErr = 999;
for ic = 1:length(allC)
    for is = 1:length(allSigma)
        thisC = allC(ic);
        thisSigma = allSigma(is);
        model= svmTrain(X, y, thisC, @(x1, x2) gaussianKernel(x1, x2, thisSigma)); 
        predictions = svmPredict(model, Xval);
        err = mean(double(predictions ~= yval));
        fprintf('Error of C = %f, sigma = %f is %f.', thisC, thisSigma, err);
        if err < minErr
            C = thisC;
            sigma = thisSigma;
            minErr = err;
        end
    end
end
fprintf('Best parameters are C = %f, sigma = %f', C, sigma);

% =========================================================================

end
