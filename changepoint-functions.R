###############################################################################
# Forward Filtering Backward Sampling Function
###############################################################################
latent_state_FFBS <- function(y_n, theta, K = 2, transition_matrix = NULL) {
  
  # creating the emissions matrix
  emissions_matrix <- matrix(0, nrow = K, ncol = length(y_n))
  for(i in 1:K) {
    emissions_matrix[i,] <- dnorm(y_n, theta[i], sqrt(3))
  }
  
  # first, we do forward filtering
  alpha <- matrix(0, nrow = length(y_n), ncol = K)
  alpha[1,] <- emissions_matrix[,1]*rep(1/K, K)
  
  for(t in 2:length(y_n)) {
    for(j in 1:K){
      alpha[t,j] <- dnorm(y_n[t], theta[j], sqrt(3)) * sum(alpha[t - 1,] * transition_matrix[,j])
    }
    # normalization of probabilities so they sum to one
    alpha[t,] <- alpha[t,]/sum(alpha[t,]) 
    }
  
  # next we do backward sampling
  z_sample <- rep(NA, length(y_n))
  z_sample[length(y_n)] <- sample(1:K, size = 1, prob = alpha[length(y_n),])
  
  for(t in (length(y_n) - 1):1) {
    prob_t <- c()
    j <- z_sample[t+1] 
    for(i in 1:K) {
      prob_t[i] <- alpha[t, i] * transition_matrix[i,j]/sum(alpha[t,] * transition_matrix[,j])
      }
    z_sample[t] <- sample(1:K, size = 1, prob = prob_t)
  }
  return(z_sample)
}

###############################################################################
# Gibbs Sampling Function
###############################################################################

gibbs_sampler <- function(y_n, transition_matrix, alpha = matrix(1, K, K),
                          K = 2, niter = 2000) {
  
  # initial parameter values
  mu <- mean(y_n)
  v_squared <- rinvgamma(1,1.5,1.5)
  theta <- rnorm(K, mu, sqrt(v_squared))
  
  tau_list <- list()
  for (iter in 1:niter) {
  
  # sampling hidden state sequence
  z_sample <- latent_state_FFBS(y_n = y_n, theta = theta,
                                transition_matrix = transition_matrix)
  
  # update the transition matrix
  counts <- matrix(0, nrow = K, ncol = K)
  for (t in 1:(length(z_sample) - 1)) {
    i <- z_sample[t]
    j <- z_sample[t + 1]
    counts[i, j] <- counts[i, j] + 1
  }
  
  for (k in 1:K) {
    transition_matrix[k, ] <- rdirichlet(1, alpha[k, ] + counts[k, ])
  }
  
  tau <- which(diff(z_sample) != 0) + 1
  
  # update the theta estimates
  y_tilde <- c()
  for (i in 1:K) {
    y_k <- y_n[z_sample == i]
    y_tilde[i] <- if_else(length(y_k) > 0, mean(y_k), 0)
  }
  
  theta <- c()
  for(i in 1:K) {
    n_i <- length(y_n[z_sample == i])
    theta_mean <- ((n_i*y_tilde[i])/3 + mu/v_squared)*(1/(n_i/3 + 1/v_squared))
    theta_sd <- sqrt(1/(n_i/3 + 1/v_squared))
    theta <- c(theta, rnorm(1, theta_mean, theta_sd))
  }
  
  # update the mu estimates (assuming non-informative prior)
  mu <- rnorm(1, mean(theta), sqrt(v_squared/(K)))
  
 # update the v_squared estimates
  v_squared <- rinvgamma(1, shape = 1.5 + (K)/2, scale = 1.5 + (1/2) * sum((theta - mu)^2))
  
  # enforce order to prevent label switching:
  ordering <- order(theta)
  theta <- theta[ordering]
  z_sample <- match(z_sample, ordering)
  
  tau_list <- append(tau_list, list(list(z_sample, theta, tau)))
  }
  return(tau_list)
}