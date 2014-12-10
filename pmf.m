function constrained_pmf(dataFileName, mapSparseData)

load(dataFileName)

training_size = size(train_vec, 1);
batch_size = 100000;
batch_count = floor(training_size / batch_size);


if mapSparseData
  map_to_user_ids = unique([train_vec(:,1); probe_vec(:,1)]);
  map_to_movie_ids = unique([train_vec(:,2); probe_vec(:,2)]);
  fprintf('Mapping user data to have smaller dimensions...');
  for i=1:numel(map_to_user_ids)
    train_vec(train_vec(:, 1) == map_to_user_ids(i), 1) = i;
    probe_vec(probe_vec(:, 1) == map_to_user_ids(i), 1) = i;
  end
  fprintf('Done.\n');
  fprintf('Mapping movie data to have smaller dimensions...\n');
  for j=1:numel(map_to_movie_ids)
    train_vec(train_vec(:, 2) == map_to_movie_ids(j), 2) = j;
    probe_vec(probe_vec(:, 2) == map_to_movie_ids(j), 2) = j;
  end
  fprintf('Done.\n');
end

n = numel(map_to_user_ids);   % user count
m = numel(map_to_movie_ids);  % movie count
d = 30;                       % number of features
K = 5;                        % max rating value


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TUNE-ABLE PARAMETERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

alpha = 0.5;                % gradient descent step size
lambda = 0.002;             % regularization parameter
number_of_iterations = 15;  % maximum number of iterations

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

U = zeros(d, n);    % user factor matrix
V = zeros(d, m);    % movie factor matrix

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
    shared_coefficient = -(ratings - ratings_predicted);
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
  training_RMSE = calculateRMSE(ratings, ratings_predicted);
  
  % Calculate test error
  test_size = size(probe_vec, 1);
  user_indices = probe_vec(:, 1);
  movie_indices = probe_vec(:, 2);
  ratings = probe_vec(:, 3);
  
  ratings = mapRatings(ratings, K);
  [U, ~] = calculateUserCoefficientMatrix(d, n, Y, W, user_indices, movie_indices);
  ratings_predicted = calculatePredictedRatings(U, V(:, movie_indices));
    
  ratings = unmapRatings(ratings, K);
  ratings_predicted = unmapRatings(ratings_predicted, K);
  test_RMSE = calculateRMSE(ratings, ratings_predicted);
  
  fprintf('Training RMSE: %2.3f\n', training_RMSE);
  fprintf('Test RMSE: %2.3f\n', test_RMSE);
end
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
end

function RMSE = calculateRMSE(ratings, ratings_predicted)
  RMSE = sqrt(sum((ratings - ratings_predicted).^2) / numel(ratings));
end
