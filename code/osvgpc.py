
from __future__ import absolute_import
import tensorflow as tf
import numpy as np
from gpflow.param import Param, DataHolder
from gpflow.model import GPModel
from gpflow import transforms, conditionals, kullback_leiblers
from gpflow.mean_functions import Zero
from gpflow._settings import settings
from gpflow.minibatch import MinibatchData
float_type = settings.dtypes.float_type


class OSVGPC(GPModel):
    """
    Online Sparse Variational GP classification.

    Streaming Gaussian process approximations
    Thang D. Bui, Cuong V. Nguyen, Richard E. Turner
    NIPS 2017
    """

    def __init__(self, X, Y, kern, likelihood, mu_old, Su_old, Kaa_old, Z_old, Z, mean_function=Zero(),
                 num_latent=None, q_diag=False, whiten=True, minibatch_size=None):

        # sort out the X, Y into MiniBatch objects.
        if minibatch_size is None:
            minibatch_size = X.shape[0]
        self.num_data = X.shape[0]
        X = MinibatchData(X, minibatch_size, np.random.RandomState(0))
        Y = MinibatchData(Y, minibatch_size, np.random.RandomState(0))

        # init the super class, accept args
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        self.q_diag, self.whiten = q_diag, whiten
        self.Z = Param(Z)
        self.num_latent = num_latent or Y.shape[1]
        self.num_inducing = Z.shape[0]

        # init variational parameters
        self.q_mu = Param(np.zeros((self.num_inducing, self.num_latent)))
        if self.q_diag:
            self.q_sqrt = Param(np.ones((self.num_inducing, self.num_latent)),
                                transforms.positive)
        else:
            q_sqrt = np.array([np.eye(self.num_inducing)
                               for _ in range(self.num_latent)]).swapaxes(0, 2)
            # , transforms.LowerTriangular(q_sqrt.shape[2]))  # Temp remove transform
            self.q_sqrt = Param(q_sqrt)

        self.mu_old = DataHolder(mu_old, on_shape_change='pass')
        self.M_old = Z_old.shape[0]
        self.Su_old = DataHolder(Su_old, on_shape_change='pass')
        self.Kaa_old = DataHolder(Kaa_old, on_shape_change='pass')
        self.Z_old = DataHolder(Z_old, on_shape_change='pass')

    def build_prior_KL(self):
        if self.whiten:
            if self.q_diag:
                KL = kullback_leiblers.gauss_kl_white_diag(
                    self.q_mu, self.q_sqrt)
            else:
                KL = kullback_leiblers.gauss_kl_white(self.q_mu, self.q_sqrt)
        else:
            K = self.kern.K(self.Z) + tf.eye(self.num_inducing,
                                             dtype=float_type) * settings.numerics.jitter_level
            if self.q_diag:
                KL = kullback_leiblers.gauss_kl_diag(self.q_mu, self.q_sqrt, K)
            else:
                KL = kullback_leiblers.gauss_kl(self.q_mu, self.q_sqrt, K)
        return KL

    def build_correction_term(self):
        # TODO
        Mb = tf.shape(self.Z)[0]
        Ma = self.M_old
        # jitter = settings.numerics.jitter_level
        jitter = 1e-4
        Saa = self.Su_old
        ma = self.mu_old
        obj = 0
        # a is old inducing points, b is new
        mu, Sigma = self.build_predict(self.Z_old, full_cov=True)
        Sigma = Sigma[:, :, 0]
        Smm = Sigma + tf.matmul(mu, tf.transpose(mu))
        Kaa = self.Kaa_old + np.eye(Ma) * jitter
        LSa = tf.cholesky(Saa)
        LKa = tf.cholesky(Kaa)
        obj += tf.reduce_sum(tf.log(tf.diag_part(LKa)))
        obj += - tf.reduce_sum(tf.log(tf.diag_part(LSa)))

        Sainv_ma = tf.matrix_solve(Saa, ma)
        obj += -0.5 * tf.reduce_sum(ma * Sainv_ma)
        obj += tf.reduce_sum(mu * Sainv_ma)

        Sainv_Smm = tf.matrix_solve(Saa, Smm)
        Kainv_Smm = tf.matrix_solve(Kaa, Smm)
        obj += -0.5 * tf.reduce_sum(tf.diag_part(Sainv_Smm) - tf.diag_part(Kainv_Smm))
        return obj

    def build_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        """

        # Get prior KL.
        KL = self.build_prior_KL()

        # Get conditionals
        fmean, fvar = self.build_predict(self.X, full_cov=False)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, settings.dtypes.float_type) /\
            tf.cast(tf.shape(self.X)[0], settings.dtypes.float_type)

        # compute online correction term
        online_reg = self.build_correction_term()

        return tf.reduce_sum(var_exp) * scale - KL + online_reg

    def build_predict(self, Xnew, full_cov=False):
        mu, var = conditionals.conditional(Xnew, self.Z, self.kern, self.q_mu,
                                           q_sqrt=self.q_sqrt, full_cov=full_cov, whiten=self.whiten)
        return mu + self.mean_function(Xnew), var
