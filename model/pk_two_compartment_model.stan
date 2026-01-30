// Set up the two compartment ODE


functions{

  vector pk_two_compartment(
    real t, 
    vector u, 
    real k_a, 
    real CL, 
    real V_cent,
    
    // Add these parameters compared to the one compartment model
    real Q,
    real V_peri
    ){
    
    // u(t) = (u_gut(t), u_cent(t), u_peri(t))
    real u_gut = u[1];
    real u_cent = u[2];
    real u_peri = u[3];
    
    // du_dt = (u'_gut(t), u'_cent(t), u'_peri(t)) 
    vector[3] du_dt;
    du_dt[1] = -k_a * u_gut;
    du_dt[2] = k_a * u_gut - (CL / V_cent + Q / V_cent) * u_cent + (Q / V_peri) * u_peri;
    du_dt[3] = (Q / V_cent) * u_cent - (Q / V_peri) * u_peri;
    
    return du_dt;
    
  }
}





data{

  int<lower=1> n_obs;                                  // Number of observations
  array[n_obs] real<lower=0> t;                           // Time points (hours)
  array[n_obs] real<lower=0> y;                // Observed concentrations (mg/L)
  real t0;                                                       // Initial time
  vector[3] u0;           // Initial conditions (u_gut(0), u_cent(0), u_peri(0))
  
}



parameters{

  // All paramters are non-negative
  
  real<lower=0> k_a;                                          // Absorption rate
  real<lower=0> CL;                                     // Elimination clearance
  real<lower=0> V_cent;                     // Volume of the central compartment
  real<lower=0> sigma;                   // Measurement model standard deviation
  
  real<lower=0> Q;                               // Intercompartmental clearance
  real<lower=0> V_peri;                  // Volume of the peripheral compartment

}





transformed parameters {

  // Solve ODE system using Runge-Kutta method
  
  array[n_obs] vector[3] u = ode_rk45(
    pk_two_compartment, 
    u0, t0, 
    t, k_a, CL, V_cent, Q, V_peri
    );
  
  
  
  // Calculate predicted concentrations 
  // c(t) = u_cent(t) / V_cent
  // u = (u_gut, u_cent, u_peri) so we want the second component of u
  
  vector[n_obs] c_hat;
  for (i in 1:n_obs) {
    c_hat[i] = u[i][2] / V_cent;
  }
}




model {

  // Priors (Equation 4)
  CL ~ lognormal(log(10), 0.25);
  V_cent ~ lognormal(log(35), 0.25);
  k_a ~ lognormal(log(2.5), 1);
  sigma ~ normal(0, 1);        
  
  // Priors (Equation 6)
  Q ~ lognormal(log(15), 0.5);
  V_peri ~ lognormal(log(105), 0.5);
  
  // Likelihood (Equation 3)
  y ~ lognormal(log(c_hat), sigma);
}




generated quantities {

  // Samples from the posterior predictive distribution
  vector[n_obs] y_pred;
  
  // Log-likelihood
  vector[n_obs] log_lik;
  
  for (i in 1:n_obs) {
    y_pred[i]    = lognormal_rng(log(c_hat[i]), sigma);
    log_lik[i]   = lognormal_lpdf(y[i] | log(c_hat[i]), sigma);
  }
}



