function [U, V, RMSE] = als(R, d, maxit, tol)

[N,M]=size(R);

U=rand(d,N);
V=rand(d,M);

RMSE = [];
for n = 1:maxit
    U = (pinv(V*V')*V)*R';
    U = (U>0).*U;
    U = (U'./(repmat(sum(U'),N,1)+eps))'; 

    V = (U'*pinv(U*U'))'*R;
    V = V.*(V>0);

    RMSE = [RMSE, mean(mean(abs(R-U'*V)))/mean(mean(R))];

    if RMSE(n) < tol || n == maxit 
        break;
    end
end