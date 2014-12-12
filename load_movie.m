% movieIDs, userIDs, rating_matrix are what you need
fprintf('loading data...\n');
load moviedata_s3.mat
movieIDs = sort(unique(train_vec(:, 1)));
userIDs = sort(unique(train_vec(:, 2)));
m = size(movieIDs, 1);
n = size(userIDs, 1);
rating_matrix = zeros(m, n);

for i = 1 : m
    j = find(train_vec(:, 1) == movieIDs(i));
    urj = sortrows(train_vec(j, 2:3), 1);
    indexj = find(ismember(userIDs, urj(:, 1)));
    rating_matrix(i, indexj) = urj(:, 2);
end

p = size(probe_vec, 1);
for i = 1 : p
    probe_vec(i, 3) = rating_matrix(find(movieIDs == probe_vec(i, 1)), find(userIDs == probe_vec(i, 2)));
end

isInProbe = ismember(train_vec, probe_vec, 'rows');
train_vec = train_vec(~isInProbe, :);

st = size(train_vec, 1);
st = st - mod(st, 100000);
train_vec = train_vec(1:st, :);

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

  movie_count = j;
  user_count = i;
  batches = numel(train_vec) / 100000;

save moviedata_s3_new.mat train_vec probe_vec movie_count user_count batches

fprintf('done loading data...\n');