from matplotlib.animation import FuncAnimation
from matplotlib import animation, rc
import matplotlib.pyplot as plt
import numpy as np

def create_static_figure(sol, guesses, true_vals, derivs, t, updates_to_plot, display_on=True):
    '''
    Create main figure of solution and errors
    '''
    x, y, xt, yt = sol.y
    alpha,  beta,  delta,  gamma  = true_vals
    alphas, betas, deltas, gammas = guesses
    dxdt, dydt, dxtdt, dytdt      = derivs.T

    alpha_update, beta_update, delta_update, gamma_update = updates_to_plot

    trace       = delta + gamma
    determinant = delta * gamma - alpha * beta

    u = abs(x - xt)
    v = abs(y - yt)
    ut = abs(dxdt - dxtdt)
    vt = abs(dydt - dytdt)

    relu  = abs(x - xt)/abs(x)
    relv  = abs(y - yt)/abs(y)
    relut = abs(dxdt - dxtdt)/abs(dxdt)
    relvt = abs(dydt - dytdt)/abs(dydt)

    alpha_err = abs(alpha - alphas)
    beta_err  = abs(beta - betas)
    delta_err = abs(delta - deltas)
    gamma_err = abs(gamma - gammas)

    # In case you need to know at what time points the parameter guesses change
    # a_change = np.where(np.diff(alphas, prepend=alphas[0]))[0]*dt
    # b_change = np.where(np.diff(betas, prepend=betas[0]))[0]*dt

    fig,ax = plt.subplots(2,2,figsize=(10,8))
    plt.rcParams['font.size'] = '16'

    ax[0,0].plot(xt, yt, 'C0', label=r'$(\tilde{x},\tilde{y})$')
    ax[0,0].plot(x,  y,  'k',  label=r'$(x,y)$')
    ax[0,0].set_title('Plot of system')
    ax[0,0].legend()
    for label in (ax[0,0].get_xticklabels() + ax[0,0].get_yticklabels()):
        label.set_fontsize(16)
        xmin, xmax, ymin, ymax = ax[0,0].axis() # used for animation formatting

    if alpha_update:
        ax[0,1].plot(t, alpha_err, 'C1', label=r'$|\tilde{\alpha}-\alpha|$')
    if beta_update:
        ax[0,1].plot(t, beta_err,  'C2', label=r'$|\tilde{\beta}-\beta|$')
    if delta_update:
        ax[0,1].plot(t, delta_err, 'C3', label=r'$|\tilde{\delta}-\delta|$')
    if gamma_update:
        ax[0,1].plot(t, gamma_err, 'C4', label=r'$|\tilde{\gamma}-\gamma|$')
    ax[0,1].set_yscale('log')
    ax[0,1].set_ylim(10**-17,10**2)
    ax[0,1].set_title('Parameter error')
    ax[0,1].grid()
    ax[0,1].legend()
    for label in (ax[0,1].get_xticklabels() + ax[0,1].get_yticklabels()):
        label.set_fontsize(16)

    if trace <= 0 and determinant > 0: # Plot absolute errors for stable cases
        ax[1,0].plot(t, u, 'k',  label=r'$|\tilde{x} - x|$')
        ax[1,0].plot(t, v, 'C0', label=r'$|\tilde{y} - y|$')
        ax[1,0].set_yscale('log')
        ax[1,0].set_title('Position error')
        ax[1,0].grid()
        ax[1,0].legend()
        for label in (ax[1,0].get_xticklabels() + ax[1,0].get_yticklabels()):
            label.set_fontsize(16)

        ax[1,1].plot(t, ut, 'k',  label=r'$|\dot{\tilde{x}} - \dot{x}|$')
        ax[1,1].plot(t, vt, 'C0', label=r'$|\dot{\tilde{y}} - \dot{y}|$')
        ax[1,1].set_yscale('log')
        ax[1,1].set_title('Velocity error')
        ax[1,1].grid()
        ax[1,1].legend()
        for label in (ax[1,1].get_xticklabels() + ax[1,1].get_yticklabels()):
            label.set_fontsize(16)
    else: # Plot relative errors for unstable cases
        ax[1,0].plot(t, relu, 'k',  label=r'$\frac{|\tilde{x} - x|}{|x|}$')
        ax[1,0].plot(t, relv, 'C0', label=r'$\frac{|\tilde{y} - y|}{|y|}$')
        ax[1,0].set_yscale('log')
        ax[1,0].set_title('Rel. position error')
        ax[1,0].grid()
        ax[1,0].legend()
        for label in (ax[1,0].get_xticklabels() + ax[1,0].get_yticklabels()):
            label.set_fontsize(16)

        ax[1,1].plot(t, relut, 'k',  label=r'$\frac{|\dot{\tilde{x}} - \dot{x}|}{|\dot{x}|}$')
        ax[1,1].plot(t, relvt, 'C0', label=r'$\frac{|\dot{\tilde{y}} - \dot{y}|}{|\dot{y}|}$')
        ax[1,1].set_yscale('log')
        ax[1,1].set_title('Rel. velocity error')
        ax[1,1].grid()
        ax[1,1].legend()
        for label in (ax[1,1].get_xticklabels() + ax[1,1].get_yticklabels()):
            label.set_fontsize(16)

    if display_on:
        # manager = plt.get_current_fig_manager()
        # manager.full_screen_toggle()
        plt.show()
        return None
    else:
        return anim_limits

def create_system_animation(sol, t):
    '''
    Displays animation of system solutions
    '''
    x, y, xt, yt   = sol.y

    fig,ax         = plt.subplots()
    plot_objects   = []

    line_true,     = plt.plot(x[:1],y[:1],'k-')
    plot_objects  += [line_true]

    lp_true,       = plt.plot(x[0],y[0],'k.',ms=10)
    plot_objects  += [lp_true]

    line_tilde,    = plt.plot(xt[:1],yt[:1],'b-')
    plot_objects  += [line_tilde]

    lp_tilde,      = plt.plot(xt[0],yt[0],'b.',ms=10)
    plot_objects  += [lp_tilde]

    anim_limits    = create_static_figure(sol, guesses, true_vals, derivs, t, display_on=False)

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    time_label = plt.text(0.9,0.1,'t = {:.4f}'.format(t[0]),ha='center', va='center', transform=ax.transAxes)
    def animate(i):
        plot_objects[0].set_data( x[:50*i],  y[:50*i])
        plot_objects[1].set_data( x[ 50*i],  y[ 50*i])
        plot_objects[2].set_data(xt[:50*i], yt[:50*i])
        plot_objects[3].set_data(xt[ 50*i], yt[ 50*i])
        time_label.set_text('t = {:.4f}'.format(t[50*i]))
        return plot_objects,time_label

    ani = FuncAnimation(fig, animate, interval=10, frames=int(len(t)/50))
    ani.show()
