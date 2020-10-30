from julia import Main

# Top-hat evolution equation in flat GR+LCDM; for \rho = (1 + delta_NL) in [0.1, 10], 
# spherical evolution in MG with LCDM expansion follows by simply rescaling the 
# initial conditions with linear theory ratios
jul_rhs_gr = Main.eval("""
    function rhs_gr(dy,y,p,x)
        Omega_m, deltai, xi = p
        Omega_L = 1. - Omega_m
        H = sqrt(Omega_m * exp(-3*x) + Omega_L)
        H2 = Omega_m * exp(-3*x) + Omega_L
        dH = -(3*exp(-3*x) * Omega_m)/(2 * H)
        delta_NL = (1. + deltai) / (exp(xi-x)*y[1] + 1.)^3 - 1.
    
        dy[1] = y[2]
        dy[2] = -(dH/H)*y[2] - 0.5 * ((Omega_m * exp(-3*x) - 2. * Omega_L)/H2) * y[1] - 0.5 * Omega_m * exp(-3*x) / H2 * (exp(x-xi) + y[1]) * delta_NL
    end
    """)

# add you favourite spherical evolution here...