functions {
     /** 
     * ODE for the inverse Hubble parameter. 
     * System State E is 1 dimensional.  
     * The system has 2 parameters theta = (om, w)
     * 
     * where 
     * 
     *   om:       dark matter energy density 
     *   w:        dark energy equation of state parameter
     *
     * The system redshift derivative is 
     * 
     * d.E[1] / d.z  =  
     *  1.0/sqrt(om * pow(1+z,3) + (1-om) * (1+z)^(3 * (1+w)))
     * 
     * @param z redshift at which derivatives are evaluated. 
     * @param E system state at which derivatives are evaluated. 
     * @param params parameters for system. 
     */ 
     vector Ez(real z,
               vector H,
               vector params
               ) {
           vector[1] dEdz;
           real p1;
           real p2;
           p1 = params[1];
           p2 = params[2];
           dEdz[1] = 1.0/sqrt(p1*(1+z)^3
                     +(1-p1)*(1+z)^(3*(1+p2)));
           return dEdz;
    } 
}
data {
    int<lower=1> nobs;              // number of data points
    vector[1] E0;                     // integral(1/H) at z=0                           
    real z0;                        // initial redshift, 0
    real c;                         // speed of light
    real H0;                        // hubble parameter
    vector[nobs] mu;                  // distance modulus
    array[nobs] real muerr;             // error in distance modulus
    array[nobs] real z;          // redshift
    real ompri;                     // mean of om gaussian  prior
    real dompri;                    // std on om gaussian prior
    real wmin;                      // min flat w prior
    real wmax;                      // max flat w prior
}
parameters{
      real<lower=0, upper=1> om;       // dark matter energy density
      real<lower=wmin, upper=wmax> w;  // dark energy equation of state parameter
      real M;
}
transformed parameters{
      array[nobs] vector[1] DC;                        // co-moving distance 
      vector[2] pars;                           // ODE input = (om, w)
      vector[nobs] dl;                          // luminosity distance
      real DH;                                // Hubble distance = c/H0
 
 
      DH = (c/H0);
      pars[1] = om;
      pars[2] = w;
      
      // Integral of 1/E(z) 
      DC = ode_rk45(Ez, E0, z0, z, pars);//,  x_r, x_i);
      for (i in 1:nobs) {
            dl[i] = M + 5 * log10(DH * (1 + z[i]) * DC[i, 1]);
      }
}
model{
     // priors
     om ~ normal(ompri, dompri);
     w ~ uniform(wmin, wmax);
     M ~ normal(0, 50);
  
     // likelihhod
     mu ~ normal(dl, muerr);
}
