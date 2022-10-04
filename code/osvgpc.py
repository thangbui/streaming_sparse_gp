import tensorflow as tf
import numpy as np
import gpflow
from gpflow import Parameter, default_float
from gpflow import conditionals, kullback_leiblers
from gpflow.inducing_variables import InducingPoints
from gpflow.models import GPModel, InternalDataTrainingLossMixin
from gpflow.utilities import positive, triangular
from packaging import version  # required to handle GPflow breaking changes


class OSVGPC(GPModel, InternalDataTrainingLossMixin):
    """
    Online Sparse Variational GP classification.

    Streaming Gaussian process approximations
    Thang D. Bui, Cuong V. Nguyen, Richard E. Turner
    NIPS 2017
    """

    def __init__(self, data, kernel, likelihood, mu_old, Su_old, Kaa_old, Z_old, Z, mean_function=None,
                 q_diag=False, whiten=True):

        self.data = gpflow.models.util.data_input_to_tensor(data)
        # self.num_data = X.shape[0]
        self.num_data = None

        # init the super class, accept args
        num_latent_gps = GPModel.calc_num_latent_gps_from_data(data, kernel, likelihood)
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)

        self.q_diag, self.whiten = q_diag, whiten
        self.inducing_variable = InducingPoints(Z)
        num_inducing = self.inducing_variable.num_inducing

        # init variational parameters
        q_mu = np.zeros((num_inducing, self.num_latent_gps))
        self.q_mu = Parameter(q_mu, dtype=default_float())  # [M, P]

        if q_diag:
            ones = np.ones(
                (num_inducing, self.num_latent_gps), dtype=default_float()
            )
            self.q_sqrt = Parameter(ones, transform=positive())  # [M, P]
        else:
            np_q_sqrt = np.array(
                [
                    np.eye(num_inducing, dtype=default_float())
                    for _ in range(self.num_latent_gps)
                ]
            )
            self.q_sqrt = Parameter(np_q_sqrt, transform=triangular())  # [P, M, M]

        self.mu_old = tf.Variable(mu_old, shape=tf.TensorShape(None), trainable=False)
        self.M_old = Z_old.shape[0]
        self.Su_old = tf.Variable(Su_old, shape=tf.TensorShape(None), trainable=False)
        self.Kaa_old = tf.Variable(Kaa_old, shape=tf.TensorShape(None), trainable=False)
        self.Z_old = tf.Variable(Z_old, shape=tf.TensorShape(None), trainable=False)

    def prior_kl(self):
        return kullback_leiblers.prior_kl(self.inducing_variable, self.kernel, self.q_mu, self.q_sqrt, whiten=self.whiten)

    def correction_term(self):
        Mb = self.inducing_variable.num_inducing
        Ma = self.M_old
        # jitter = gpflow.default_jitter()
        jitter = gpflow.utilities.to_default_float(1e-4)
        Saa = self.Su_old
        ma = self.mu_old
        # a is old inducing points, b is new
        mu, Sigma = self.predict_f(self.Z_old, full_cov=True)
        Sigma = tf.squeeze(Sigma, axis=0)
        Smm = Sigma + tf.matmul(mu, mu, transpose_b=True)
        Kaa = gpflow.utilities.add_noise_cov(self.Kaa_old, jitter)
        LSa = tf.linalg.cholesky(Saa)
        LKa = tf.linalg.cholesky(Kaa)
        obj = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LKa)))
        obj += - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LSa)))

        Sainv_ma = tf.linalg.cholesky_solve(LSa, ma)
        obj += -0.5 * tf.reduce_sum(ma * Sainv_ma)
        obj += tf.reduce_sum(mu * Sainv_ma)

        Sainv_Smm = tf.linalg.cholesky_solve(LSa, Smm)
        Kainv_Smm = tf.linalg.cholesky_solve(LKa, Smm)
        obj += -0.5 * tf.reduce_sum(tf.linalg.diag_part(Sainv_Smm) - tf.linalg.diag_part(Kainv_Smm))
        return obj

    def maximum_log_likelihood_objective(self) -> tf.Tensor:  # type: ignore
        return self.elbo()

    def elbo(self):
        """
        This gives a variational bound on the model likelihood.
        """
        X, Y = self.data

        # Get prior KL.
        kl = self.prior_kl()

        # Get conditionals
        fmean, fvar = self.predict_f(X, full_cov=False)

        # Get variational expectations.
        if version.parse(gpflow.__version__) < version.Version("2.6.0"):
            var_exp = self.likelihood.variational_expectations(fmean, fvar, Y)
        else:
            # breaking change https://github.com/GPflow/GPflow/pull/1919
            var_exp = self.likelihood.variational_expectations(X, fmean, fvar, Y)

        # re-scale for minibatch size
        if self.num_data is not None:
            raise NotImplementedError("need to update code to ExternalDataTrainingLossMixin")
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)

        # compute online correction term
        online_reg = self.correction_term()

        return tf.reduce_sum(var_exp) * scale - kl + online_reg

    def predict_f(self, Xnew, full_cov=False, full_output_cov=False):
        mu, var = conditionals.conditional(Xnew, self.inducing_variable, self.kernel, self.q_mu,
                                           q_sqrt=self.q_sqrt, full_cov=full_cov, white=self.whiten,
                                           full_output_cov=full_output_cov)
        return mu + self.mean_function(Xnew), var
