from scipy.integrate import solve_ivp
from src.figures import create_static_figure
from typing import Optional
import numpy as np
import random
import math
import time


class Simulation:
    def __init__(self, **kwargs):

        # Set up system
        if not ("trace" in kwargs or "det" in kwargs):
            if not (all(key in kwargs for key in ["alpha", "beta", "delta", "gamma"])):
                raise ValueError("Insufficient parameters provided")
            else:
                self.alpha = kwargs["alpha"]
                self.beta = kwargs["beta"]
                self.delta = kwargs["delta"]
                self.gamma = kwargs["gamma"]
        else:
            self.trace = kwargs["trace"]
            self.det = kwargs["det"]
            self.delta, self.alpha, self.beta, self.gamma = (
                self.generate_integer_matrix()
            )
        
        # True system parameters
        self.true_vals = np.array([self.alpha, self.beta, self.delta, self.gamma])

        # Set up initial guesses according to which parameters you want to estimate
        # All other parameters are assumed known
        to_update = kwargs["updates"]
        initial_guesses = np.array([self.alpha, self.beta, self.delta, self.gamma])
        initial_guesses[to_update] += 10
        self.alpha0, self.beta0, self.delta0, self.gamma0 = initial_guesses

        self.updates_to_plot = list(self.true_vals != initial_guesses)
        self.updates_on = self.true_vals != initial_guesses

        # Nudging parameters
        # NOTE: Full nudging matrix not advised. Diagonal entries seem to work best but
        #       could play around with off-diagonal entries. I just use a multiple of
        #       identity matrix
        mu1 = mu4 = kwargs["mu"]
        mu2 = mu3 = 0
        self.mus = np.array([mu1, mu2, mu3, mu4])

        # Relaxation period (time to wait between updates)
        self.relax = kwargs["relax"]
        self.time_between = np.array([self.relax] * 4)

        # Position, velocity thresholds for updates
        # NOTE: Right now we are not using any thresholds, they are all set to infinity
        #       with no decay
        self.u_thold = np.inf
        self.v_thold = np.inf
        self.ut_thold = np.inf
        self.vt_thold = np.inf
        self.d = 1
        self.thresholds = np.array(
            [self.u_thold, self.v_thold, self.ut_thold, self.vt_thold, self.d]
        )

        # Initialize system
        self.S0 = np.array([1, 1, 3, 3])

        self.guesses = [[self.alpha0], [self.beta0], [self.delta0], [self.gamma0]]
        self.parms = np.array(
            [
                self.alpha,
                self.beta,
                self.delta,
                self.gamma,
                self.alpha0,
                self.beta0,
                self.delta0,
                self.gamma0,
            ],
            dtype=np.float64,
        )
        self.derivs = [self.calc_rhs(self.S0, self.mus, self.parms)]
        self.err = [[abs(self.S0[2] - self.S0[0]), abs(self.S0[3] - self.S0[1])]]

        self.last_updates = np.array([0, 0, 0, 0])
        self.idx_last_updates = np.array([0, 0, 0, 0])
        self.tfe = []

        self.rule = "drop_deriv"

        # Simulation parameters
        self.sim_time = 100
        self.dt = 0.0001
        self.t_span = [0, self.sim_time]
        self.t = np.arange(0, self.sim_time, self.dt)

    def generate_integer_matrix(self):
        """
        Generates an integer-valued 2x2 matrix with given trace and determinant.
        A = [[a,b],[c,d]].
        This is almost certainly not the best way to do this, and the parameters
        do not need to be integer-valued, but I thought it would be easier to work
        with for preliminary experiments.
        You can toggle the range in np.random.randint() to have more variety in the matrices
        generated, but the parameter values get very large very quickly
        """
        div = []
        while not div:
            a = np.random.randint(5)
            d = self.trace - a
            b_mult = a * d - self.det
            for val in range(1, int(math.sqrt(abs(b_mult)) + 1)):
                if b_mult % val == 0:
                    div.extend([val, int(b_mult / val), -val, -int(b_mult / val)])
        b = random.choice(div)
        c = int((a * d - self.det) / b)

        return a, b, c, d

    def calc_rhs(self, S, mus, parms):
        """
        Returns derivatives of x, y, xt, yt
        """

        # Unpack inputs
        x, y, xt, yt = S
        mu1, mu2, mu3, mu4 = mus
        alpha, beta, delta, gamma, alphat, betat, deltat, gammat = parms
        
        # Derivative formulas
        rhs = [
            delta * x + alpha * y,
            beta * x + gamma * y,
            deltat * xt + alphat * yt - mu1 * (xt - x) - mu2 * (yt - y),
            betat * xt + gammat * yt - mu3 * (xt - x) - mu4 * (yt - y),
        ]
        return rhs

    def get_tholds(self, pos_err, ilast):
        """
        Determines update threshold on position error using log linear fit of data
        since last update
        NOTE: No thresholding is being used in current version of algorithm, but this
        is in place in case it needs to be used in future iterations
        """
        log_pos_err = np.log(pos_err[:, ilast:])
        xcoords = np.arange(log_pos_err.shape[1])
        ycoords = log_pos_err.T
        p = np.polyfit(xcoords, ycoords, 1)
        u_thold = np.exp(p[0, 0] * xcoords[-1] + p[1, 0])
        v_thold = np.exp(p[0, 1] * xcoords[-1] + p[1, 1])
        return u_thold, v_thold

    def update_formula(self, S, errors):
        """
        Returns relevant parameter update formulas
        """

        # Unpack inputs
        x, y, xt, yt = S
        u, v, ut, vt = errors
        mu1, mu2, mu3, mu4 = self.mus
        alpha, beta, delta, gamma, alphat, betat, deltat, gammat = self.parms

        # drop_deriv is the default algorithm to use
        if self.rule == "drop_deriv":
            return [
                (deltat * xt + alphat * yt - delta * x - mu1 * u - mu2 * v) / y,
                (betat * xt + gammat * yt - gamma * y - mu3 * v - mu4 * v) / x,
                (deltat * xt + alphat * yt - alpha * y - mu1 * u - mu2 * v) / x,
                (betat * xt + gammat * yt - beta * x - mu3 * v - mu4 * v) / y,
            ]
        
        # Exact update formula for dev purposes only
        if self.rule == "exact":
            return [
                (deltat * xt + alphat * yt - delta * x - mu1 * u - mu2 * v - ut) / y,
                (betat * xt + gammat * yt - gamma * y - mu3 * u - mu4 * v - vt) / x,
                (deltat * xt + alphat * yt - alpha * y - mu1 * u - mu2 * v - ut) / x,
                (betat * xt + gammat * yt - beta * x - mu3 * u - mu4 * v - vt) / y,
            ]

        # No update
        return [alphat, betat, deltat, gammat]

    def update_parms(self, update_idx, S, pos_err):
        """
        Updates parameter guesses [if thresholds are met]
        """

        curr_time = self.tfe[-1]

        # Unpack input
        x, y, xt, yt = S
        u_thold, v_thold, ut_thold, vt_thold, d = self.thresholds

        # Calculate derivatives and errors
        dxdt, dydt, dxtdt, dytdt = self.calc_rhs(S, self.mus, self.parms)
        u = xt - x
        v = yt - y
        ut = dxtdt - dxdt
        vt = dytdt - dydt
        errors = np.array([u, v, ut, vt])

        # Determine if thresholds are met (they always are in this implementation)
        if abs(u) / abs(x) <= u_thold and abs(v) / abs(y) <= v_thold:
            for i in update_idx:
                new = self.update_formula(S, errors)[i]
                self.parms[i + 4] = new
                self.guesses[i].append(new)
                self.last_updates[i] = curr_time
                self.idx_last_updates[i] = pos_err.shape[1] - 1
        else:
            for i in update_idx:
                self.guesses[i].append(self.parms[i + 4])

        return

    def model(self, t, S):
        """
        Function called by solve_ivp() to return the time derivative of
        the system S at time t while dynamically updating parameters
        """

        # Record time of function call in tfe
        curr_time = t
        self.tfe.append(curr_time)

        # Current x error |xt-x| and y error |yt-y|
        unow = abs(S[2] - S[0])
        vnow = abs(S[3] - S[1])
        self.err.append([unow, vnow])
        pos_err = np.array(self.err).T

        # Check if relaxation period has lapsed
        time_since = curr_time - self.last_updates
        time_threshold = time_since > self.time_between

        # Threshold on relative poisition error for stopping updates
        stop_update = 1e-10
        if unow / abs(S[0]) < stop_update and vnow / abs(S[1]) < stop_update:
            self.updates_on[0] = self.updates_on[1] = self.updates_on[2] = self.updates_on[3] = False

        to_update = np.logical_and(time_threshold, self.updates_on)

        update_idx = [i for i, x in enumerate(to_update) if x]

        # Make updates
        self.update_parms(update_idx, S, pos_err)

        no_update_idx = [i for i, x in enumerate(to_update) if not x]
        for i in no_update_idx:
            self.guesses[i].append(self.parms[i + 4])

        St = self.calc_rhs(S, self.mus, self.parms)
        self.derivs.append(St)

        return St

    def run(self):
        # Run simulation
        start = time.time()
        sol = solve_ivp(
            self.model, t_span=self.t_span, y0=self.S0, method="BDF", t_eval=self.t
        )

        # Handle outputs
        guesses = np.array(self.guesses, dtype=np.float64)
        derivs = np.array(self.derivs)

        # Reshape guessea dn derivs arrays to match solution
        # This is necessary because solve_ivp() calls model() minimally to improve computational efficiency
        # but returns sol, an interpolation of the solution at all time points of t_eval. However, guesses
        # and derivs are only recorded when model() is called (which is why we recorded tfe :) )
        num_iter = round(self.tfe[-1] / self.dt)
        t_sol = np.linspace(0, self.dt * num_iter, num=num_iter)
        f_eval = np.searchsorted(self.tfe, t_sol)
        guesses = guesses[:, f_eval]
        derivs = derivs[f_eval, :]

        alphas, betas, deltas, gammas = guesses
        alpha, beta, delta, gamma = self.true_vals

        print(
            f"Runtime: {time.time()-start} seconds.\nFinal alpha error: {abs(alpha-alphas)[-1]}\nFinal beta error: {abs(beta-betas)[-1]}\nFinal delta error: {abs(delta-deltas)[-1]}\nFinal gamma error: {abs(gamma-gammas)[-1]}"
        )

        # Generate figure
        create_static_figure(
            sol, guesses, self.true_vals, derivs, self.t, self.updates_to_plot
        )
        return sol, guesses, derivs
