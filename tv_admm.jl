function tv_admm(X, Y, M, mu, rho, nIters)
#
# Syntax:       X = inpaint_admm(Y, M, mu, rho, nIters)
#
# Inputs:       Y is an m x n matrix
#
#               M is an m x n matrix such that
#
#                   M[i, j] = {1, if Y[i, j] is not corrupted
#                             {0, if Y[i, j] is corrupted
#
#               mu is the regularization parameter
#
#               rho is the ADMM parameter
#
#               tau is the step size to use, and should satisfy
#               tau ≤ 1 / (1 + 8 * gamma)
#               nIters is the number of iterations to perform
#
# Outputs:      X is the inpainted m x n matrix
#
# Description: Solves the TV-regularized inpainting problem
#
#               min_{x} 0.5 \|y - Ax\|ˆ2 + \mu \|Cx\|_1
#
#               via ADMM. In the above, y = Y[:], x = X[:], A is the
#               inpainting matrix, and C is the 2D
#                first differences matrix
(m, n) = size(Y);
C = [kron(D(n), speye(m));  kron(speye(n), D(m))]
A = spdiagm(M[:])

x = X[:]
y = Y[:]
# z =
u = zeros(2 * length(y))
z = soft(C*x + u, mu/rho)
u = u + C*x - z
P = A' * A + rho * C' * C
Aty = A' * y

# ADMM updates
Xall = zeros(m, n, nIters)
cost = zeros(nIters)
for k = 1:nIters
    # x update
    x = P \ (Aty + rho * C' * (z - u))
    # z update
    z = soft(C*x + u, mu/rho)
    # u update
    u = u + C*x - z

    # Save stats
    Xall[:, :, k] = reshape(x, m, n)
    cost[k] = 0.5 * norm(y - A * x, 2)^2 + mu * norm(C * x, 1)
end

return Xall, cost
end

function soft(z, mu)
    return sign(z) .* max(abs(z) - mu, 0)
end

function D(n)
    Dn = spdiagm([-ones(n), ones(n - 1)], [0, 1], n, n)
    Dn[n, 1] = 1
    return Dn
end
