# import matplotlib
# matplotlib.use('Agg')

import numpy as np
import matplotlib as mpl
# mpl.use('pgf')


def figsize(scale):
    # Get this from LaTeX using \the\textwidth
    fig_width_pt = 397.4849
    inches_per_pt = 1.0 / 72.27                       # Convert pt to inch
    # Aesthetic ratio (you could change this)
    golden_mean = (np.sqrt(5.0) - 1.0) / 6
    fig_width = fig_width_pt * inches_per_pt * scale    # width in inches
    fig_height = fig_width * golden_mean              # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    # blank entries should cause plots to inherit fonts from the document
    "font.serif": [],
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 7,               # LaTeX default is 10pt font.
    "font.size": 7,
    "legend.fontsize": 6,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
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

# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib
# accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

import gpflow
import osvgpc
import matplotlib.pyplot as plt


def init_Z(cur_Z, new_X, use_old_Z=True, first_batch=True):
    if use_old_Z:
        Z = np.copy(cur_Z)
    else:
        M = cur_Z.shape[0]
        M_old = int(0.8 * M)
        M_new = M - M_old
        old_Z = cur_Z[np.random.permutation(M)[0:M_old], :]
        new_Z = new_X[np.random.permutation(new_X.shape[0])[0:M_new], :]
        Z = np.vstack((old_Z, new_Z))
    return Z


def get_data(iid):
    X = np.loadtxt('../data/banana_train_x.txt', delimiter=',')
    y = np.loadtxt('../data/banana_train_y.txt', delimiter=',')
    y = y.reshape((y.shape[0], 1))
    y[y==-1] = 0
    Xtest = np.loadtxt('../data/banana_test_x.txt', delimiter=',')
    ytest = np.loadtxt('../data/banana_test_y.txt', delimiter=',')
    ytest = ytest.reshape((ytest.shape[0], 1))
    ytest[ytest==-1] = 0
    if not iid:
        # todo sort
        idxs = np.argsort(X[:, 0])
        X = X[idxs, :]
        y = y[idxs, :]
    return X, y, Xtest, ytest


def gridParams():
    mins = [-3.2, -2.5]
    maxs = [3.2, 2.5]
    nGrid = 50
    xspaced = np.linspace(mins[0], maxs[0], nGrid)
    yspaced = np.linspace(mins[1], maxs[1], nGrid)
    xx, yy = np.meshgrid(xspaced, yspaced)
    Xplot = np.vstack((xx.flatten(), yy.flatten())).T
    return mins, maxs, xx, yy, Xplot


def plot_model(ax, m, cur_x, cur_y, seen_x=None, seen_y=None, test_x=None, test_y=None):
    col1 = '#0172B2'
    col2 = '#CC6600'
    mins, maxs, xx, yy, Xplot = gridParams()
    # p = m.predict_y(Xplot)[0]
    mf, vf = m.predict_f(Xplot)
    mf = mf.numpy()
    vf = vf.numpy()
    ax.plot(
        cur_x[:, 0][cur_y[:, 0] == 1],
        cur_x[:, 1][cur_y[:, 0] == 1],
        'o', color=col1, mew=0, alpha=0.6)
    ax.plot(
        cur_x[:, 0][cur_y[:, 0] == 0],
        cur_x[:, 1][cur_y[:, 0] == 0],
        'o', color=col2, mew=0, alpha=0.6)
    if seen_x is not None:
        ax.plot(
            seen_x[:, 0][seen_y[:, 0] == 1],
            seen_x[:, 1][seen_y[:, 0] == 1],
            'o', color=col1, mew=0, alpha=0.05)
        ax.plot(
            seen_x[:, 0][seen_y[:, 0] == 0],
            seen_x[:, 1][seen_y[:, 0] == 0],
            'o', color=col2, mew=0, alpha=0.05)
    if hasattr(m, 'inducing_variable'):
        Z = m.inducing_variable.Z
        ax.plot(Z[:, 0], Z[:, 1], 'ko', mew=0, ms=3, alpha=0.8)
    ax.contour(xx, yy, mf.reshape(*xx.shape),
                [0], colors='k', linewidths=1.4, zorder=100)
    # plt.contour(xx, yy, p.reshape(*xx.shape), [0.5],
    #             colors='k', linewidths=1.8, zorder=100)
    if test_x is not None:
        mf, _ = m.predict_f(test_x)
        mf = mf.numpy()
        pred_y = 1.0 * (mf > 0)
        err = np.sum(np.abs(pred_y - test_y)) / mf.shape[0]
        ax.set_title('error=%.2f'%err)


def run_vfe(no_batches, M, use_old_Z, iid):

    X, y, Xtest, ytest = get_data(iid)
    N = X.shape[0]
    mb_size = int(np.floor(N / no_batches))
    fig, axs = plt.subplots(1, no_batches+1, figsize=figsize(1), sharey=True)
    maxiter = 2000
    for i in range(no_batches):
        Xi = X[i * mb_size:(i + 1) * mb_size, :]
        yi = y[i * mb_size:(i + 1) * mb_size, :]
        if i == 0:
            Z1 = Xi[np.random.permutation(Xi.shape[0])[0:M], :]
            model = gpflow.models.SVGP(gpflow.kernels.RBF(lengthscales=np.ones(2)),
                                       gpflow.likelihoods.Bernoulli(), Z1)
            gpflow.optimizers.Scipy().minimize(
                model.training_loss_closure((Xi, yi)), model.trainable_variables,
                options=dict(disp=1, maxiter=maxiter))
        else:
            Zinit = init_Z(Zopt, Xi, use_old_Z)
            model = osvgpc.OSVGPC((Xi, yi), gpflow.kernels.RBF(lengthscales=np.ones(2)),
                                  gpflow.likelihoods.Bernoulli(),
                                  mu, Su, Kaa, Zopt, Zinit)
            gpflow.optimizers.Scipy().minimize(
                model.training_loss, model.trainable_variables,
                options=dict(disp=1, maxiter=maxiter))
        Zopt = model.inducing_variable.Z.numpy()
        mu, Su = model.predict_f(Zopt, full_cov=True)
        if len(Su.shape) == 3:
            Su = Su[0, :, :] + 1e-4 * np.eye(mu.shape[0])
        Kaa = model.kernel(model.inducing_variable.Z)
        if i == 0:
            seen_x = None
            seen_y = None
        else:
            seen_x = X[:i * mb_size, :]
            seen_y = y[:i * mb_size, :]
        plot_model(axs[i], model, Xi, yi, seen_x, seen_y, Xtest, ytest)

    # run sparse GP
    Z = X[np.random.permutation(X.shape[0])[0:M], :]
    model = gpflow.models.SVGP(gpflow.kernels.RBF(lengthscales=np.ones(2)),
                               gpflow.likelihoods.Bernoulli(), Z)
    gpflow.optimizers.Scipy().minimize(
        model.training_loss_closure((X, y)), model.trainable_variables,
        options=dict(disp=1, maxiter=maxiter))
    plot_model(axs[-1], model, X, y, None, None, Xtest, ytest)

    for i in range(no_batches+1):
        axs[i].locator_params(nbins=5, axis='y')
        axs[i].locator_params(nbins=5, axis='x')
        axs[i].tick_params('both', length=3, width=0.5, which='minor')
        axs[i].tick_params('both', length=3, width=0.5, which='major')
        axs[i].tick_params('both', length=3, width=0.5, which='minor')
        axs[i].tick_params('both', length=3, width=0.5, which='major')
        axs[i].set_xlabel(r'$x_1$')
    axs[0].set_ylabel(r'$x_2$')
    plt.subplots_adjust(wspace=0.01, hspace=0.01)

    fig.savefig('../tmp/cla_VFE_M_%d_iid_%r.png' % (M, iid), bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':

    use_old_Z = False
    no_batches = 3
    np.random.seed(42)
    M = 30

    iid = True
    run_vfe(no_batches, M, use_old_Z, iid)

    iid = False
    run_vfe(no_batches, M, use_old_Z, iid)
