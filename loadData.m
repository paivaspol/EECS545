function preference_matrix = loadData()
  movie_data = csvread('mv_sample_data.csv');
  
  n = max(movie_data(:,1)); % number of users
  m = max(movie_data(:,2)); % number of movies
  preference_matrix = zeros(n, m);
  
  for i = 1:size(movie_data, 1)
    preference_matrix(movie_data(i,1), movie_data(i,2)) = movie_data(i,3);
  end
end