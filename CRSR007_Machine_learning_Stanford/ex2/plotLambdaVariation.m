function plotData(theta, lambda, X, y)
%PLOTLAMBDAVARIATION Plots the data points X and y into a new figure  with
%the decision boundary defined by theta and different values of lambda.
%   PLOTLAMBDAVARIATION(theta,lambda,X,y) plots the data points with + for the positive examples
%   and o for the negative examples.
%   X is assumed to be a MxN matrix.
%   theta is assumed to ba a NXP matrix, where N is number of features and P is number of lambdas.
%   lambda is assumed to ba a PX1 matrix, where P is number of different lambda values.

% Plot Data
plotData(X(:,2:3), y);

% Labels and Legend
xlabel('Microchip Test 1');
ylabel('Microchip Test 2');

% Specified in plot order
legend('y = 1', 'y = 0');

hold on;

% Here is the grid range
u = linspace(-1, 1.5, 50);
v = linspace(-1, 1.5, 50);
[U, V] = meshgrid(u, v);
z = zeros(length(u), length(v), length(lambda));
% Evaluate z = theta*x over the grid
for k = 1:length(lambda)
    for i = 1:length(u)
        for j = 1:length(v)
            z(i,j,k) = mapFeature(u(i), v(j))*theta(:,k);
        end
    end
    z(:,:,k) = z(:,:,k)';
    contour(U, V, z(:,:,k), [0, 0], 'LineWidth', 2);
end

title('Regularized logistic regression with different lambda');

% Labels and Legend
xlabel('Microchip Test 1');
ylabel('Microchip Test 2');

legend('y = 1', 'y = 0', 'Decision boundary');
hold off;

% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);

fprintf('\nProgram paused. Press enter to continue.\n');
hold off;

end
