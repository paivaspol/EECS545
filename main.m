R = loadData();
[U, S, V] = svd(R);
variance = var(R);
variance_u = var(U);
variance_v = var(V);
lambda_u = variance / variance_u;
lambda_v = variance / variance_v;
[U, V] = pmf_gradient_descent(R, U, V, lambda_u, lambda_v, size(R, 1), size(R, 2), 30);