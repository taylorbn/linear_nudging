
from scipy.integrate import solve_ivp
import numpy as np
import figures
import random
import time
import math

def generate_integer_matrix(tr, det):
    '''
    Generates an integer-valued 2x2 matrix with given trace and determinant.
    A = [[a,b],[c,d]].
    This is almost certainly not the best way to do this, and the parameters
    do not need to be integer-valued, but I thought it would be easier to work
    with for preliminary experiments.
    You can toggle the range in np.random.randint() to have more variety in the matrices
    generated, but the parameter values get very large very quickly
    '''
    div = []
    while not div:
        a = np.random.randint(5)
        d   = tr - a
        b_mult = a*d - det
        for val in range(1, int(math.sqrt(abs(b_mult))+1)):
            if b_mult % val == 0:
                div.extend([val, int(b_mult / val), -val, -int(b_mult/val)])
    b = random.choice(div)
    c = int((a*d - det)/b)
    if a+d != tr or a*d-b*c != det:
        print('ERROR: something went wrong in matrix generation')
        exit()
    else:
        return a, b, c, d

def calc_rhs(S, mus, parms):
    '''
    Returns derivatives of x, y, xt, yt
    '''
    # extract input
    x, y, xt, yt                  = S
    mu1, mu2, mu3, mu4            = mus
    alpha,  beta,  delta,  gamma, alphat, betat, deltat, gammat = parms

    # derivative formulae
    rhs = [delta  * x + alpha * y,
           beta   * x + gamma * y,
           deltat * xt + alphat * yt - mu1 * (xt - x) - mu2 * (yt - y),
           betat  * xt + gammat * yt - mu3 * (xt - x) - mu4 * (yt - y)
          ]
    return rhs

def rule_string(which):
    '''
    Converts rule index to string for retrieving update formula from dictionary
    '''
    switcher = {
        0: 'constant',
        1: 'exact',
        2: 'drop_deriv'
    }
    return switcher.get(which)

def update_formula(rule, S, errors, derivs_now, mus, parms):
    '''
    Returns parameter update formulas
    '''
    # handle input
    x, y, xt, yt             = S
    u, v, ut, vt             = errors
    dxdt, dydt, dxtdt, dytdt = derivs_now
    mu1, mu2, mu3, mu4       = mus
    alpha,  beta, delta, gamma, alphat, betat, deltat, gammat   = parms

    switcher = {
        'constant'  : [alphat, betat, deltat, gammat],
        'exact'     : [(deltat*xt + alphat*yt - delta*x - mu1*u - mu2*v - ut)/y,
                       (betat*xt  + gammat*yt - gamma*y - mu3*u - mu4*v - vt)/x,
                       (deltat*xt + alphat*yt - alpha*y - mu1*u - mu2*v - ut)/x,
                       (betat*xt  + gammat*yt - beta*x  - mu3*u - mu4*v - vt)/y],
        'drop_deriv': [(deltat*xt + alphat*yt - delta*x - mu1*u - mu2*v)/y,
                       (betat*xt  + gammat*yt - gamma*y - mu3*v - mu4*v)/x,
                       (deltat*xt + alphat*yt - alpha*y - mu1*u - mu2*v)/x,
                       (betat*xt  + gammat*yt - beta*x  - mu3*v - mu4*v)/y],
    }
    return switcher.get(rule, "Update rule not recognized")

def get_tholds(pos_err, ilast):
    '''
    Determines update threshold on position error using log linear fit of data
    since last update
    NOTE: No thresholding is being used in current version of algorithm, but this
          is in place in case it needs to be used in future iterations
    '''
    log_pos_err = np.log(pos_err[:,ilast:])
    xcoords = np.arange(log_pos_err.shape[1])
    ycoords = log_pos_err.T
    p = np.polyfit(xcoords,ycoords,1)
    u_thold = np.exp(p[0,0]*xcoords[-1] + p[1,0])
    v_thold = np.exp(p[0,1]*xcoords[-1] + p[1,1])
    return u_thold, v_thold

def update_parms(which, S, mus, parms, guesses, thresholds, rule, time_tracker, pos_err, ilast):
    '''
    Updates parameter guesses [if thresholds are met]
    '''
    curr_time = time_tracker[-1][-1]
    # Extract input
    x, y, xt, yt                            = S
    mu1, mu2, mu3, mu4                      = mus
    alpha, beta, delta, gamma, alphat, betat, deltat, gammat = parms
    u_thold, v_thold, ut_thold, vt_thold, d = thresholds

    # Calculate derivatives
    dxdt, dydt, dxtdt, dytdt   = calc_rhs(S, mus, parms)
    derivs_now                 = np.array([dxdt, dydt, dxtdt, dytdt])

    # Calculate error
    u      = xt - x
    v      = yt - y
    ut     = dxtdt - dxdt
    vt     = dytdt - dydt
    errors = np.array([u, v, ut, vt])

    if abs(u)/abs(x) <= u_thold and abs(v)/abs(y) <= v_thold:
        for i in which:
            new      = update_formula(rule, S, errors, derivs_now, mus, parms)[i]
            parms[i+4] = new
            guesses[i].append(new)
            time_tracker[0][i]  = curr_time
            idx_last_updates[i] = pos_err.shape[1] - 1
    else:
        for i in update_idx:
            parm_idx = i + 4
            guesses[i].append(parm_idx)

def model(t, S, mus, parms, thresholds, derivs, guesses, time_tracker, updates_on, err, idx_last_updates):
    '''
    function called by odeint to return the time derivative of
    the system S at time t.

    INPUT
        mus              : NumPy array of shape (4) with values [mu1, mu2, mu3, mu4]
                           Note a full nudging matrix is not advised (typically choose
                           diagonal or off diagonal elements depending on which params
                           you want to recover)
        parms            : NumPy array of shape (8) with values [alpha, beta, delta, gamma,
                           alphat, betat, deltat, gammat]
                           where alpha, beta, delta, gamma are the true parameter values and
                           alphat, betat, deltat, gammat are the current guesses.
        thresholds       : NumPy array of shape(5) with values [u_thold,
                           ut_thold, v_thold, vt_thold, d] which are used to decide whether to
                           update parameters. Note these are not being used right now, updates
                           are being made any time the relaxation period has lapsed
        derivs           : list of time derivatives at all function evaluations
        guesses          : list of parameter guesses made at all function evals
        time_tracker     : list containing last_updates (the timestamp of the last parameter updates)
                                       time_between (relaxation time)
                                       tfe (times model() is called in simulation)
        updates_on       : boolean array specifying which parameters should be updated
        err              : local position errors (for get_tholds() -- not in use right now)
        idx_last_updates : index of last updates (for get_tholds() -- not in use right now)
    '''
    # Unpack args
    alpha,  beta, delta, gamma, alphat, betat, deltat, gammat   = parms
    last_updates, time_between, tfe = time_tracker

    curr_time = t
    time_tracker[-1].append(curr_time)  # Record time of function call in tfe
    unow = abs(S[2]-S[0])               # Current x error |xt-x|
    vnow = abs(S[3]-S[1])               # Current y error |yt-y|
    err.append([unow,vnow])
    pos_err = np.array(err).T

    time_since     = curr_time  - last_updates
    time_threshold = time_since > time_between # Check if relaxation time has lapsed for parameters

    stop_update = 1e-10                 # Threshold on relative position error for stopping updates
    if unow/abs(S[0]) < stop_update and vnow/abs(S[1]) < stop_update:
      updates_on[0] = updates_on[1] = updates_on[2] = updates_on[3] = False

    # Update parms where relaxation period has lapsed and stop criteria not met
    to_update     = np.logical_and(time_threshold, updates_on)

    update_idx    = [i for i,x in enumerate(to_update) if x]
    update_parms(update_idx, S, mus, parms, guesses, thresholds, rule, time_tracker, pos_err, idx_last_updates)

    no_update_idx = [i for i,x in enumerate(to_update) if not x]
    for i in no_update_idx:
        guesses[i].append(parms[i+4])

    St = calc_rhs(S, mus, parms)
    derivs.append(St)

    return St
def run_simulation(t_span, S0, t, true_vals, args):
    mus, parms, thresholds, derivs, guesses, time_tracker, updates_on, err, idx_last_updates = args
    # ---------- Run simulation ------------------------
    start        = time.time()
    sol          = solve_ivp(model,
                             t_span = t_span,
                             y0     = S0,
                             method ='BDF',
                             t_eval = t,
                             args   = args
                            )

    # ---------- Handle output ------------------------
    guesses      = np.array(guesses, dtype=np.float64)
    derivs       = np.array(derivs)

    # Reshape guesses and derivs arrays to match solution
    # This is necessary because solve_ivp() calls model() minimally to improve computational efficiency
    # but returns sol, an interpolation of the solution at all time points of t_eval. However, guesses
    # and derivs are only recorded when model() is called (which is why we recorded tfe :) )
    num_iter     = round(tfe[-1]/dt)
    t_sol        = np.linspace(0,dt*num_iter, num=num_iter)
    f_eval       = np.searchsorted(tfe, t_sol)
    guesses      = guesses[:,f_eval]
    derivs       = derivs[f_eval,:]

    alphas, betas, deltas, gammas = guesses
    alpha,  beta,  delta,  gamma  = true_vals
    print('Runtime: {:.4f} seconds. \nFinal alpha error: {:.4e} \nFinal beta error: {:.4e} \nFinal delta error: {:.4e} \nFinal gamma error: {:.4e}'.format(time.time()-start,
                                                                                                                                                           abs(alpha-alphas)[-1],
                                                                                                                                                           abs(beta-betas)[-1],
                                                                                                                                                           abs(delta-deltas)[-1],
                                                                                                                                                           abs(gamma-gammas)[-1]))
    return sol, guesses, derivs

if __name__ == '__main__':
    # Run simulation
    import sys
    # Read command line inputs (Code to do this within the script instead commented out below)
    if   len(sys.argv) == 7:
        str_inputs = sys.argv[1:]
        int_inputs = [int(i) for i in str_inputs]
        delta, alpha, beta, gamma, mu_val, relax = int_inputs
    elif len(sys.argv) == 5:
        str_inputs = sys.argv[1:]
        int_inputs = [int(i) for i in str_inputs]
        tr, det, mu_val, relax = int_inputs
        delta, alpha, beta, gamma = generate_integer_matrix(tr, det)

    rule_to_use = 2  # rules = {0:'constant', 1:'exact', 2:'drop_deriv'}
    rule        = rule_string(rule_to_use)

    # ------------ Equation parameters -----------------
    # True system parameters
    # tr = 0
    # det = 2
    # delta, alpha, beta, gamma = generate_integer_matrix(tr,det)
    # alpha =  2
    # beta  = -2
    # delta = -1
    # gamma = 0

    true_vals = np.array([alpha, beta, delta, gamma])

    # Initial guesses for alpha, beta, delta, gamma
    # NOTE: Code is set up to only update parameters with guesses different from
    #       the true value. This needs to be toggled within the script now but could
    #       be implemented from the command line if you wanted
    alpha0          = alpha
    beta0           = beta
    delta0          = delta + 10
    gamma0          = gamma + 10
    initial_guesses = np.array([alpha0, beta0, delta0, gamma0])

    updates_to_plot = list(true_vals != initial_guesses)

    updates_on      = true_vals != initial_guesses

    # ------------ Algorithm parameters ----------------
    # Nudging parameters
    # NOTE: Full nudging matrix not advised. Diagonal entries seem to work best but
    #       could play around with off diagonal entries. I typically use multiple of
    #       identity matrix
    # mu_val    = 800
    mu1 = mu4 = mu_val
    mu2 = mu3 = 0

    # Relaxation period (time to wait between updates)
    # relax        = 4
    time_between = np.array([relax] * 4)

    # Position, velocity thresholds for updates
    # NOTE: Right now we are not using any thresholds, they are all set to infinity
    #       with no decay
    u_thold    = np.inf
    v_thold    = np.inf
    ut_thold   = np.inf
    vt_thold   = np.inf
    d          = 1      # Threshold decay factor

    # Package mus, parms, thresholds for use in solve_ivp()
    mus        = np.array([mu1, mu2, mu3, mu4])
    parms      = np.array([alpha, beta, delta, gamma, alpha0, beta0, delta0, gamma0], dtype=np.float64)
    thresholds = np.array([u_thold, v_thold, ut_thold, vt_thold, d])

    # ------------ Simulation parameters ----------------
    sim_time   = 100                        # Stopping time
    dt         =   0.0001                   # Timestep
    t_span     = [0,sim_time]
    t          = np.arange(0, sim_time, dt)

    # ------------ Initialize system --------------------
    S0         = np.array([1, 1,3, 3])                 # Initialize [x, y, xt, yt]
    guesses    = [[alpha0],[beta0],[delta0],[gamma0]]  # Guesses
    derivs     = [calc_rhs(S0,mus,parms)]              # Derivatives
    err        = [[abs(S0[2]-S0[0]),abs(S0[3]-S0[1])]] # Position error

    last_updates     = np.array([0,0,0,0])             # Time of last updates
    idx_last_updates = np.array([0,0,0,0])             # Index of last updates
    tfe = []                                           # To record times when model() is called

    # Package into time_tracker arg for solve_ivp()
    time_tracker = [last_updates, time_between, tfe]

    args = (mus, parms, thresholds, derivs, guesses, time_tracker, updates_on, err, idx_last_updates)

    sol, guesses, derivs = run_simulation(t_span, S0, t, true_vals, args)

    figures.create_static_figure(sol, guesses, true_vals, derivs, t, updates_to_plot)
