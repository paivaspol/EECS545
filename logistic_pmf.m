function logistic_pmf(dataFileName, mapSparseData)

load moviedata_s1_new

n = user_count;
m = movie_count;
batch_count = batches;
batch_size = 100000;
d = 30;                       % number of features
K = 5;                        % max rating value


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TUNE-ABLE PARAMETERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

alpha = 0.02;                % gradient descent step size
lambda = 0.002;             % regularization parameter
number_of_iterations = 50;  % maximum number of iterations

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

U = 0.1*randn(d, user_count);
V = 0.1*randn(d, movie_count);

for iterations = 1:number_of_iterations
  for batch=1:batch_count
    fprintf('Iteration %d, Batch %d\n', iterations, batch);
    start_index = ((batch - 1) * batch_size) + 1;
    end_index = batch * batch_size;
    user_indices = train_vec(start_index:end_index, 1);
    movie_indices = train_vec(start_index:end_index, 2);
    ratings = train_vec(start_index:end_index, 3);

    ratings = mapRatings(ratings, K);
    ratings_predicted = calculatePredictedRatings(U(:,user_indices), V(:, movie_indices));
    
    % Calculate gradients
    shared_coefficient = -(ratings - ratings_predicted) .* ratings_predicted .* (1 - ratings_predicted);
    repeat_shared_coefficient = repmat(shared_coefficient, 1, d)';
    % not sure about these gradients...
    U_gradient = repeat_shared_coefficient .* V(:, movie_indices) + lambda * U(:, user_indices);
    V_gradient = repeat_shared_coefficient .* U(:, user_indices) + lambda * V(:, movie_indices);
    
    delta_U = zeros(size(U));
    delta_V = zeros(size(V));
    
    for i = 1:batch_size
      delta_U(:, user_indices(i)) = delta_U(:, user_indices(i)) + U_gradient(:, i);
      delta_V(:, movie_indices(i)) = delta_V(:, movie_indices(i)) + V_gradient(:, i);
    end
    
    U = U - alpha * delta_U;
    V = V - alpha * delta_V;
  end
    
  % Calculate training error
  ratings = unmapRatings(ratings, K);
  ratings_predicted = unmapRatings(ratings_predicted, K);
  training_RMSE(iterations) = calculateRMSE(ratings, ratings_predicted);
  
  % Calculate test error
  test_size = size(probe_vec, 1);
  user_indices = probe_vec(:, 1);
  movie_indices = probe_vec(:, 2);
  ratings = probe_vec(:, 3);
  
  ratings = mapRatings(ratings, K);
  ratings_predicted = calculatePredictedRatings(U(:,user_indices), V(:, movie_indices));
    
  ratings = unmapRatings(ratings, K);
  ratings_predicted = unmapRatings(ratings_predicted, K);
  test_RMSE(iterations) = calculateRMSE(ratings, ratings_predicted);
  
  fprintf('Training RMSE: %2.3f\n', training_RMSE(iterations));
  fprintf('Test RMSE: %2.3f\n', test_RMSE(iterations));
  save logistic_s1_30 training_RMSE test_RMSE
end

figure(1);
plot (1:number_of_iterations, training_RMSE);
figure(2);
plot (1:number_of_iterations, test_RMSE);

end

function result = mapRatings(ratings, K)
  result = zeros(size(ratings));
  for i=1:size(ratings)
    result(i) = (ratings(i) - 1)/(K - 1);
  end
end

function result = unmapRatings(ratings, K)
  result = zeros(size(ratings));
  for i=1:size(ratings)
    result(i) = (ratings(i)* (K - 1)) + 1;
    if result(i) > 5
      result(i) = 5;
    elseif result(i) < 1
      result(i) = 1;
    end
  end
end

function ratings_predicted = calculatePredictedRatings(U, V)
  ratings_predicted = sum(U .* V, 1)';
  ratings_predicted = (1 + exp(-ratings_predicted)) .^ -1;
end

function RMSE = calculateRMSE(ratings, ratings_predicted)
  RMSE = sqrt(sum((ratings - ratings_predicted).^2) / numel(ratings));
end
