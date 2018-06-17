function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

y_pred = sum(theta'.*X, 2) % this does the theta vector times each X_j 
%vector and then sums along the axis to get sum(theta_j*x_j) for each row

J = sum(y_pred - y).^2/(2*m)

% =========================================================================

end
