%% A gradient descent algorithm for finding the optimal U and V.
function [U, V] = pmf_gradient_descent(R, U, V, lambda_u, lambda_v, m, n, d)

  % constants for the function
  epsilon = 1e-4;
  alpha = 10;
  
  U_new = U;
  V_new = V;
  
  old_obj_fn = pmf_objective_function(R, U_new, V_new, lambda_u, lambda_v);
  
  while 1    
    
    % Perform gradient descent. There are two updates to find U and V.
    % update U
    for i = 1:n
      % find gradient
      running_sum = zeros([size(v,2), 1]);
      for j = 1:m
        if R(i,j) > 0
          running_sum = running_sum + (R(i,j) - U(:,i)' * V(:,j)) * (-V(:,j));
        end
      end
      % update u(t+1)
      U_new(:,i) = U_new + (alpha .* (running_sum  + (lambda_u .* U(:,i))));
    end
    
    % update V
    for j = 1:m
      % find gradient
      running_sum = zeros([size(v,2), 1]);
      for i = 1:n
        if R(i, j) > 0
          running_sum = running_sum + (R(i,j) - U(:,i)' * V(:,j)) * (-U(:,i));
        end
      end
      % update u(t+1)
      U_new(:,i) = U_new + (alpha .* (running_sum  + (lambda_v .* V(:,j))));
    end
    
    % Compute Objective Function values and update the variables.
    obj_fn = pmf_objective_function(I, R, U_new, V_new, lambda_u, lambda_v);
    U = U_new;
    V = V_new;
    
    if abs(old_obj_fn - obj_fn) <= epsilon
      break;
    end
    old_obj_fn = obj_fn;
  end
end

%% Computes the objective function for the pmf case.
function obj_fn = pmf_objective_function(R, U, V, lambda_u, lambda_v)
  reg_u = lambda_u * 0.5 * norm(U)^2;
  reg_v = lambda_v * 0.5 * norm(V)^2;
  running_sum = 0;
  for i = 1:size(U,2)
    for j = 1:size(V,2)
      if R(i,j) > 0
        running_sum = running_sum + (R(i,j) - U(:,i)'*V(:,j))^2;
      end
    end
  end
  obj_fn = running_sum + reg_u + reg_v;
end
