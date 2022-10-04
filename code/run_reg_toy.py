# import matplotlib
# matplotlib.use('Agg')

import numpy as np
import matplotlib as mpl
# mpl.use('pgf')

def figsize(scale, ratio=None):
    fig_width_pt = 397.4849                         # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/4           # Aesthetic ratio (you could change this)
    if ratio is not None:
        golden_mean = ratio
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    # blank entries should cause plots to inherit fonts from the document
    "font.serif": [],
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(0.9),     # default fig size of 0.9 textwidth
    "pgf.preamble": [
        # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[utf8x]{inputenc}",
        # plots will be generated using this preamble
        r"\usepackage[T1]{fontenc}",
        # plots will be generated using this preamble
        r"\usepackage{amsmath}",
    ]
}
mpl.rcParams.update(pgf_with_latex)

grey = '#808080'

mpl.rcParams['axes.linewidth'] = 0.3
mpl.rcParams['axes.edgecolor'] = grey
mpl.rcParams['xtick.color'] = grey
mpl.rcParams['ytick.color'] = grey
mpl.rcParams['axes.labelcolor'] = "black"

import gpflow
import osgpr
import sgpr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def init_Z(cur_Z, new_X, use_old_Z=True, first_batch=True):
    if use_old_Z:
        Z = np.copy(cur_Z)
    else:
        M = cur_Z.shape[0]
        M_old = int(0.7 * M)
        M_new = M - M_old
        old_Z = cur_Z[np.random.permutation(M)[0:M_old], :]
        new_Z = new_X[np.random.permutation(new_X.shape[0])[0:M_new], :]
        Z = np.vstack((old_Z, new_Z))
    return Z


def plot_model(model, ax, cur_x, cur_y, pred_x, seen_x=None, seen_y=None):
    mx, vx = model.predict_f(pred_x)
    Zopt = model.inducing_variable.Z.numpy()
    mu, Su = model.predict_f(Zopt, full_cov=True)
    if len(Su.shape) == 3:
        Su = Su[0, :, :]
        vx = vx[:, 0]
    ax.plot(cur_x, cur_y, 'kx', mew=1, alpha=0.8)
    if seen_x is not None:
        ax.plot(seen_x, seen_y, 'kx', mew=1, alpha=0.2)
    ax.plot(pred_x, mx, 'b', lw=2)
    ax.fill_between(
        pred_x[:, 0], mx[:, 0] -  2*np.sqrt(vx), 
        mx[:, 0] +  2*np.sqrt(vx), 
        color='b', alpha=0.3)
    ax.plot(Zopt, mu, 'ro', mew=1)
    ax.set_ylim([-2.4, 2])
    ax.set_xlim([np.min(pred_x), np.max(pred_x)])
    plt.subplots_adjust(hspace = .08)
    ax.set_ylabel('y')
    ax.yaxis.set_ticks(np.arange(-2, 3, 1))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
    return mu, Su, Zopt


def get_data(shuffle):
    X = np.loadtxt('../data/reg_toy_x.txt', delimiter=',')
    y = np.loadtxt('../data/reg_toy_y.txt', delimiter=',')
    X = X.reshape(X.shape[0], 1)
    y = y.reshape((y.shape[0], 1))
    N = y.shape[0]
    gap = N//3
    X[:gap, :] = X[:gap, :] - 1
    X[2*gap:3*gap, :] = X[2*gap:3*gap, :] + 1
    if shuffle:
        idxs = np.random.permutation(N)
        X = X[idxs, :]
        y = y[idxs, :]
    return X, y


def optimize(model, **options):
    gpflow.optimizers.Scipy().minimize(
        model.training_loss, model.trainable_variables,
        options=options)


def plot_PEP_optimized(M, alpha, use_old_Z, shuffle):
    fig, axs = plt.subplots(4, 1, figsize=figsize(1, ratio=12.0/19.0), sharey=True, sharex=True)

    X, y = get_data(shuffle)

    N = X.shape[0]
    gap = N//3
    # get the first portion and call sparse GP regression
    X1 = X[:gap, :]
    y1 = y[:gap, :]
    seen_x = None
    seen_y = None
    # Z1 = np.random.rand(M, 1)*L
    Z1 = X1[np.random.permutation(X1.shape[0])[0:M], :]

    model1 = sgpr.SGPR_PEP((X1, y1), gpflow.kernels.RBF(variance=1.0, lengthscales=0.8), Z=Z1, alpha=alpha)
    model1.likelihood.variance.assign(0.001)
    optimize(model1, disp=1)

    # plot prediction
    xx = np.linspace(-2, 12, 100)[:,None]
    mu1, Su1, Zopt = plot_model(model1, axs[0], X1, y1, xx, seen_x, seen_y)
    
    # now call online method on the second portion of the data
    X2 = X[gap:2*gap, :]
    y2 = y[gap:2*gap, :]
    seen_x = X[:gap, :]
    seen_y = y[:gap, :]

    Kaa1 = model1.kernel(model1.inducing_variable.Z)

    Zinit = init_Z(Zopt, X2, use_old_Z)
    model2 = osgpr.OSGPR_PEP((X2, y2), gpflow.kernels.RBF(variance=model1.kernel.variance, lengthscales=model1.kernel.lengthscales), mu1, Su1, Kaa1, 
        Zopt, Zinit, alpha)
    model2.likelihood.variance.assign(model1.likelihood.variance)
    optimize(model2, disp=1)

    # plot prediction
    mu2, Su2, Zopt = plot_model(model2, axs[1], X2, y2, xx, seen_x, seen_y)

    # now call online method on the third portion of the data
    X3 = X[2*gap:3*gap, :]
    y3 = y[2*gap:3*gap, :]
    seen_x = np.vstack((seen_x, X2))
    seen_y = np.vstack((seen_y, y2))

    Kaa2 = model2.kernel(model2.inducing_variable.Z)

    Zinit = init_Z(Zopt, X3, use_old_Z)
    model3 = osgpr.OSGPR_PEP((X3, y3), gpflow.kernels.RBF(variance=model2.kernel.variance, lengthscales=model2.kernel.lengthscales), mu2, Su2, Kaa2, 
        Zopt, Zinit, alpha)
    model3.likelihood.variance.assign(model2.likelihood.variance)
    optimize(model3, disp=1)

    mu3, Su3, Zopt = plot_model(model3, axs[2], X3, y3, xx, seen_x, seen_y)
    axs[2].set_xlabel('x')


    Z4 = X[np.random.permutation(X.shape[0])[0:M], :]
    model4 = sgpr.SGPR_PEP((X, y), gpflow.kernels.RBF(variance=1.0, lengthscales=0.8), Z=Z4, alpha=alpha)
    model4.likelihood.variance.assign(0.001)
    optimize(model4, disp=1)

    # plot prediction
    xx = np.linspace(-2, 12, 100)[:,None]
    mu4, Su4, Zopt = plot_model(model4, axs[3], X, y, xx, None, None)
    axs[3].set_xlabel('x')

    fig.savefig('../tmp/reg_PEP_alpha_%.3f_M_%d_iid_%r.png' % (alpha, M, shuffle), bbox_inches='tight')


def plot_VFE_optimized(M, use_old_Z, shuffle):

    fig, axs = plt.subplots(4, 1, figsize=figsize(1, ratio=12.0/19.0), sharey=True, sharex=True)

    X, y = get_data(shuffle)

    N = X.shape[0]
    gap = N//3

    # get the first portion and call sparse GP regression
    X1 = X[:gap, :]
    y1 = y[:gap, :]
    seen_x = None
    seen_y = None
    # Z1 = np.random.rand(M, 1)*L
    Z1 = X1[np.random.permutation(X1.shape[0])[0:M], :]

    model1 = gpflow.models.SGPR((X1, y1), gpflow.kernels.RBF(variance=1.0, lengthscales=0.8), inducing_variable=Z1, noise_variance=0.001)
    optimize(model1, disp=1)

    # plot prediction
    xx = np.linspace(-2, 12, 100)[:,None]
    mu1, Su1, Zopt = plot_model(model1, axs[0], X1, y1, xx, seen_x, seen_y)
    
    # now call online method on the second portion of the data
    X2 = X[gap:2*gap, :]
    y2 = y[gap:2*gap, :]
    seen_x = X[:gap, :]
    seen_y = y[:gap, :]

    Kaa1 = model1.kernel(model1.inducing_variable.Z)

    Zinit = init_Z(Zopt, X2, use_old_Z)
    model2 = osgpr.OSGPR_VFE((X2, y2), gpflow.kernels.RBF(variance=model1.kernel.variance, lengthscales=model1.kernel.lengthscales), mu1, Su1, Kaa1, 
        Zopt, Zinit)
    model2.likelihood.variance.assign(model1.likelihood.variance)
    optimize(model2, disp=1)

    # plot prediction
    mu2, Su2, Zopt = plot_model(model2, axs[1], X2, y2, xx, seen_x, seen_y)

    # now call online method on the third portion of the data
    X3 = X[2*gap:3*gap, :]
    y3 = y[2*gap:3*gap, :]
    seen_x = np.vstack((seen_x, X2))
    seen_y = np.vstack((seen_y, y2))

    Kaa2 = model2.kernel(model2.inducing_variable.Z)

    Zinit = init_Z(Zopt, X3, use_old_Z)
    model3 = osgpr.OSGPR_VFE((X3, y3), gpflow.kernels.RBF(variance=model2.kernel.variance, lengthscales=model2.kernel.lengthscales), mu2, Su2, Kaa2, 
        Zopt, Zinit)
    model3.likelihood.variance.assign(model2.likelihood.variance)
    optimize(model3, disp=1)
    mu3, Su3, Zopt = plot_model(model3, axs[2], X3, y3, xx, seen_x, seen_y)
    

    Z4 = X[np.random.permutation(X.shape[0])[0:M], :]
    model4 = gpflow.models.SGPR((X, y), gpflow.kernels.RBF(variance=1.0, lengthscales=0.8), inducing_variable=Z4, noise_variance=0.001)
    optimize(model4, disp=1)

    # plot prediction
    xx = np.linspace(-2, 12, 100)[:,None]
    mu4, Su4, Zopt4 = plot_model(model4, axs[3], X, y, xx, None, None)
    axs[3].set_xlabel('x')
    fig.savefig('../tmp/reg_VFE_M_%d_iid_%r.png' % (M, shuffle), bbox_inches='tight')


if __name__ == '__main__':
    use_old_Z = False
    alpha = 0.5
    seed = 10
    shuffle = False

    np.random.seed(seed)
    plot_PEP_optimized(10, alpha, use_old_Z, shuffle)
    np.random.seed(seed)
    plot_VFE_optimized(10, use_old_Z, shuffle)

