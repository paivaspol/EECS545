%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Implementation of Probabilistic Matrix Factorization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set the constants
alpha = 50; % learning rate
lambda = 0.01; % regularization parameter
momentum = 0.8; % momentum for adaptive gradient descent

max_iterations = 5;

load moviedata
mean_rating = mean(train_vec(:,3));

training_size = length(train_vec);
test_size = length(probe_vec);

batches = 9;
movie_count = 3952;
user_count = 6040;
feature_count = 10;

triples_per_batch = 100000;

% initializes the variables
movie_features = 0.1*randn(movie_count, feature_count);
user_features = 0.1*randn(user_count, feature_count);
movie_features_inc = zeros(movie_count, feature_count);
user_features_inc = zeros(user_count, feature_count);

for iteration = 1:max_iterations

  % train over all the training data.
  for batch = 1:batches
    fprintf('Batch %d, Iteration %d\n', batch, iteration);
    user_vector = double(train_vec((batch-1)*triples_per_batch+1:batch*triples_per_batch,1));
    movie_vector = double(train_vec((batch-1)*triples_per_batch+1:batch*triples_per_batch,2));
    rating_vector = double(train_vec((batch-1)*triples_per_batch+1:batch*triples_per_batch,3));

    rating_vector = rating_vector - mean_rating;

    % Compute predictions
    predictions = sum(movie_features(movie_vector,:) .* user_features(user_vector,:),2);
    obj_fn = sum((predictions - rating_vector).^2 + 0.5*lambda*( sum(movie_features(movie_vector,:).^2 + user_features(user_vector,:).^2,2 )));

    % Compute the gradients
    first_term = repmat(2*(predictions - rating_vector),1,feature_count);
    movie_gradient = first_term .* user_features(user_vector,:)  + lambda*movie_features(movie_vector,:);
    user_gradient = first_term .* movie_features(movie_vector,:) + lambda*user_features(user_vector,:);

    dw1_movie = zeros(movie_count,feature_count);
    dw1_user = zeros(user_count,feature_count);

    for i=1:triples_per_batch
      dw1_movie(movie_vector(i),:) = dw1_movie(movie_vector(i),:) + movie_gradient(i,:);
      dw1_user(user_vector(i),:) = dw1_user(user_vector(i),:) + user_gradient(i,:);
    end

    % Update movie and user features
    movie_features_inc = momentum*movie_features_inc + alpha*dw1_movie/triples_per_batch;
    movie_features = movie_features - movie_features_inc;
    user_features_inc = momentum*user_features_inc + alpha*dw1_user/triples_per_batch;
    user_features = user_features - user_features_inc;
  end

  % Compute predictions after parameter updates
  predictions = sum(movie_features(movie_vector,:) .* user_features(user_vector,:),2);
  obj_fn = sum((predictions - rating_vector).^2 + 0.5*lambda*( sum(movie_features(movie_vector,:).^2 + user_features(user_vector,:).^2,2 )));
  train_error(iteration) = sqrt(obj_fn/triples_per_batch);

  % Compute predictions on validation set
  user_vector = double(probe_vec(:,1));
  movie_vector = double(probe_vec(:,2));
  rating_vector = double(probe_vec(:,3));
  predictions = sum(movie_features(movie_vector,:) .* user_features(user_vector,:),2) + mean_rating;
  out_of_range = find(predictions>5);
  predictions(out_of_range) = 5;
  out_of_range = find(predictions<1);
  predictions(out_of_range) = 1;

  test_error(iteration) = sqrt(sum((predictions - rating_vector).^2)/test_size);
  fprintf(1, 'epoch %4i batch %4i Training RMSE %6.4f  Test RMSE %6.4f  \n', ...
              iteration, batch, train_error(iteration), test_error(iteration));
end