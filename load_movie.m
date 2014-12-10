% movieIDs, userIDs, rating_matrix are what you need
clear;clc;
load moviedata_s1.mat
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

st = size(train_vec, 1);
st = st - mod(st, 100000);
new_train_vec = train_vec(1:st, :);