#' Fit a Chain Beta-Binomial Model
#'
#' This function fits a chain beta-binomial model to estimate the secondary attack risk (SAR) 
#' using a Bayesian approach with the `rstan` package.
#'
#' @param data A data frame containing the following columns:
#'   - `y`: Number of secondary infections.
#'   - `n`: Number of contacts.
#'   - `age`: Age of the primary patient.
#'   - `vaccination_status`: Vaccination status of the primary patient (0 = unvaccinated, 1 = vaccinated).
#' @param iter Number of iterations for the MCMC sampler. Default is 2000.
#' @param chains Number of chains for the MCMC sampler. Default is 4.
#' @param warmup Number of warm-up (burn-in) iterations. Default is 1000.
#' @param thin Thinning interval for the MCMC sampler. Default is 1.
#' @param seed Seed for reproducibility. Default is 123.
#'
#' @return A list containing:
#'   - `fit`: The Stan model fit object.
#'   - `SAR_mean`: Mean secondary attack risk (SAR).
#'   - `SAR_cred_int`: 95% credible interval for SAR.
#'   - `beta_vaccination_cred_int`: 95% credible interval for the vaccination coefficient.
#' 
#' @export
fit_chain_beta_binomial <- function(data, iter = 2000, chains = 4, warmup = 1000, thin = 1, seed = 123) {
  library(rstan)
  library(arm)
  
  stan_model_code <- "
  data {
    int<lower=0> N;  // number of observations
    int<lower=0> y[N];  // number of secondary infections
    int<lower=0> n[N];  // number of contacts
    vector[N] age;  // age of the primary patient
    int vaccination_status[N];  // vaccination status of the primary patient
  }
  parameters {
    real alpha;  // intercept
    real beta_age;  // coefficient for age
    real beta_vaccination;  // coefficient for vaccination status
    real<lower=0> phi;  // overdispersion parameter
  }
  transformed parameters {
    vector[N] mu;
    vector[N] alpha_minus_mu;
    vector<lower=0>[N] alpha_param;
    vector<lower=0>[N] beta_param;
    
    for (i in 1:N) {
      mu[i] = inv_logit(alpha + beta_age * age[i] + beta_vaccination * vaccination_status[i]);
      alpha_minus_mu[i] = mu[i] * phi;
      alpha_param[i] = alpha_minus_mu[i];
      beta_param[i] = (1 - mu[i]) * phi;
    }
  }
  model {
    alpha ~ normal(0, 10);
    beta_age ~ normal(0, 10);
    beta_vaccination ~ normal(0, 10);
    phi ~ cauchy(0, 5);

    for (i in 1:N) {
      y[i] ~ beta_binomial(n[i], alpha_param[i], beta_param[i]);
    }
  }
  "
  
  # Prepare Stan data
  stan_data <- list(
    N = nrow(data),
    y = data$y,
    n = data$n,
    age = data$age,
    vaccination_status = data$vaccination_status
  )
  
  # Fit the model
  fit <- stan(
    model_code = stan_model_code,
    data = stan_data,
    iter = iter,
    chains = chains,
    warmup = warmup,
    thin = thin,
    seed = seed
  )
  
  # Extract posterior samples
  posterior_samples <- rstan::extract(fit)
  
  # Compute mu for each observation
  compute_mu <- function(alpha, beta_age, beta_vaccination, age, vaccination_status) {
    invlogit(alpha + beta_age * age + beta_vaccination * vaccination_status)
  }
  
  mu_samples <- sapply(1:nrow(data), function(i) {
    compute_mu(
      posterior_samples$alpha, 
      posterior_samples$beta_age, 
      posterior_samples$beta_vaccination, 
      data$age[i], 
      data$vaccination_status[i]
    )
  })
  
  # Calculate SAR
  sar_samples <- rowMeans(mu_samples)
  sar_mean <- mean(sar_samples)
  sar_cred_int <- quantile(sar_samples, probs = c(0.025, 0.975))
  
  # Vaccination coefficient credible interval
  beta_vaccination_cred_int <- quantile(posterior_samples$beta_vaccination, probs = c(0.025, 0.975))
  
  list(
    fit = fit,
    SAR_mean = sar_mean,
    SAR_cred_int = sar_cred_int,
    beta_vaccination_cred_int = beta_vaccination_cred_int
  )
}

####### Toy example ########

data <- data.frame(
  y = c(2, 1, 3, 0, 1),
  n = c(10, 8, 12, 6, 9),
  age = c(30, 45, 25, 60, 35),
  vaccination_status = c(1, 0, 1, 0, 1)
)

result <- fit_chain_beta_binomial(data)

cat("SAR Mean:", result$SAR_mean, "\n")
cat("SAR 95% Credible Interval:", result$SAR_cred_int, "\n")
cat("Beta Vaccination 95% Credible Interval:", result$beta_vaccination_cred_int, "\n")
