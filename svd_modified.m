%% A singular value decomposition algorithm for finding the optimal U and V.
function [U, S, V] = svd_modified(R)
% the input R is precomputed to have 0 weight for unobserved entries
% no normalization is applied
[U, S, V] = svd(R);
end

function = svd_testing(S)
end