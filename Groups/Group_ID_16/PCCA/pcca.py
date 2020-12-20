import numpy as np
inv = np.linalg.inv

def expectation_maximize(params):
  Psi = params['Psi']
  k = params['k']
  W = params['W']
  X = params['X']
  n = params['n']
  p1 = params['p1']
  p2 = params['p2']
  Sigma1 = params['Sigma1']
  Sigma2 = params['Sigma2']

  Psi_inv = inv(Psi)
  M = inv(np.eye(k) + W.T @ Psi_inv @ W)
  S = M @ W.T @ Psi_inv @ X
  A = n * M + S @ S.T

  W_new = X @ S.T @ inv(A)

  W1       = W[:p1]
  W1_new   = W_new[:p1]
  Psi1_inv = Psi_inv[:p1, :p1]
  Psi1_new = Sigma1 - Sigma1 @ Psi1_inv @ W1 @ M @ W1_new.T

  W2       = W[p1:]
  W2_new   = W_new[p1:]
  Psi2_inv = Psi_inv[p1:, p1:]
  Psi2_new = Sigma2 - Sigma2 @ Psi2_inv @ W2 @ M @ W2_new.T

  Psi_new = np.block([[Psi1_new, np.zeros((p1, p2))],
                      [np.zeros((p2, p1)), Psi2_new]])

  params['W'] = W_new
  params['Psi'] = Psi_new


def fit(params):
  Psi = params['Psi']
  iters = params['iters']
  np.linalg.cholesky(Psi)
  for _ in range(iters):
    print(_)
    expectation_maximize(params)
    Psi = params['Psi']
    np.linalg.cholesky(Psi)


def transform(params):
  X1 = params['X1']
  X2 = params['X2']
  Psi = params['Psi']
  k = params['k']
  W = params['W']
  
  X = np.hstack([X1, X2]).T
  Psi_inv = inv(Psi)
  M = inv(np.eye(k) + W.T @ Psi_inv @ W)
  Z = M @ W.T @ Psi_inv @ X
  return Z.T


def fit_transform(params):
  fit(params)
  return transform(params)


def get_projections(params, n_samples):
  Psi = params['Psi']
  k = params['k']
  W = params['W']
  X = params['X']
  p = params['p']
  p1 = params['p1']

  Psi_inv = inv(Psi)
  M = inv(np.eye(k) + W.T @ Psi_inv @ W)
  Z_post_mean = M @ W.T @ Psi_inv @ X

  X_mean = W @ Z_post_mean
  X = np.zeros((n_samples, p))

  for i in range(n_samples):
    X[i] = np.random.multivariate_normal(X_mean[:, i], Psi)

  return X[:, :p1], X[:, p1:]

def pcca(inmats, nvecs, iters = 500, nprojs = 100):
  # arguments
  # inmats : input matrices [X1,X2]
  # nvecs : number of reduced dimensions
  # iters : number of iterations for optimization
  # nprojs : number of projections to generate

  X1 = inmats[0]
  X2 = inmats[1]
  n = X1.shape[0]
  p1 = X1.shape[1]
  p2 = X2.shape[1]
  p = p1 + p2
  X = np.hstack([X1, X2]).T
  k = nvecs

  #covariances
  Sigma1 = np.cov(X1.T)
  Sigma2 = np.cov(X2.T)

  # Initialize Weights
  W1 = np.random.random((p1, k))
  W2 = np.random.random((p2, k))
  W = np.vstack([W1, W2])

  # Initialize Psi.
  prior_var1 = 1
  prior_var2 = 1
  Psi1 = prior_var1 * np.eye(p1)
  Psi2 = prior_var2 * np.eye(p2)
  Psi = np.block([[Psi1, np.zeros((p1, p2))],
                  [np.zeros((p2, p1)), Psi2]])
  
  params = {}
  params['k'] = k
  params['iters'] = iters
  params['X1'] = X1
  params['X2'] = X2
  params['n'] = n
  params['p1'] = p1
  params['p2'] = p2
  params['p'] = p
  params['X'] = X
  params['Sigma1'] = Sigma1
  params['Sigma2'] = Sigma2
  params['W'] = W
  params['Psi'] = Psi

  Z = fit_transform(params)
  X1, X2 = get_projections(params, nprojs)
  return Z,X1,X2

