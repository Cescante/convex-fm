# -*- coding: utf8 -*-
"""Convex factorization machines

Implements the solver by: Mathieu Blondel, Akinori Fujino, Naonori Ueda.
"Convex factorization machines". Proc. of ECML-PKDD 2015
http://www.mblondel.org/publications/mblondel-ecmlpkdd2015.pdf
"""

# Author: Vlad Niculae <vlad@vene.ro>
# License: Simplified BSD

# TODOS:
# * implement warm starts and regularization paths
# * options to ignore the diagonal of Z / to constrain Z to be PSD
# * implement fully corrective refitting
# * diagonal refit every K iter (requires reasonable estimate of new eigval)
# * implement projected gradient baseline for comparison

import array

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.linear_model import Ridge
from sklearn.linear_model.cd_fast import enet_coordinate_descent
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error


def _find_basis(X, residual, **kwargs):
    def _create_mv(X, residual):
        if sp.issparse(X):
            def mv(p):
                return X.T * (residual * (X * p))
        else:
            def mv(p):
                return np.dot(X.T, residual * np.dot(X, p))
        return mv

    n_features = X.shape[1]
    grad = LinearOperator((n_features, n_features),
                          matvec=_create_mv(X, residual),
                          dtype=X.dtype)
    _, p = eigsh(grad, k=1, **kwargs)

    return p.ravel()


class ConvexFM(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1., beta=1., fit_intercept=False,
                 fit_linear=False, max_iter=200,
                 max_iter_inner=100, max_rank=None, warm_start=True,
                 tol=1e-3, refit_iter=1000, eigsh_kwargs={},
                 verbose=False, random_state=0):
        """Factorization machine with nuclear norm regularization.

        minimizes 0.5 ∑(y - (b + w'x + <Z, xx'>))² + .5 * α||w||² + β||Z||_*
        Z is implicitly stored as an eigendecomposition Z = P'ΛP

        Implements the greedy coordinate descent solver from:
            Convex Factorization Machines.
            Mathieu Blondel, Akinori Fujino, Naonori Ueda.
            Proceedings of ECML-PKDD 2015
            http://www.mblondel.org/publications/mblondel-ecmlpkdd2015.pdf

        Parameters
        ----------
        alpha : float
            L2 regulariation for linear term
        beta : float,
            Nuclear (trace) norm regularization for quadratic term
        fit_intercept : bool, default: False
            Whether to fit an intercept (b). Only used if ``fit_linear=True``.
        fit_linear : bool, default: False
            Whether to fit the linear term (b + w'x).
        max_iter : int,
            Number of alternative steps in the outer loop.
        max_iter_inner : int,
            Number of iterations when solving for Z
        max_rank : int,
            Budget for the representation of Z. Default: n_features
        warm_start : bool, default: False
            Warm starts, not fully implemented yet.
        tol : bool,
            Tolerance for all subproblems.
        refit_iter : int,
            Number of iterations for diagonal refitting (Lasso)
        eigsh_kwargs : dict,
            Arguments to pass to the ARPACK eigenproblem solver. Defaults are
            ``tol=tol`` and ``maxiter=5000``.
        verbose : int,
            Degree of verbosity.
        random_state : int or np.random.RandomState,
            Random number generator (used in diagonal refitting).

        Attributes
        ----------
        ridge_ : sklearn.linear_model.Ridge instance,
            Fitted regressor for the linear part

        lams_ : list,
            Fitted eigenvalues of Z

        P_ : list,
            Fitted eigenvectors of Z

        """
        self.alpha = alpha
        self.beta = beta
        self.fit_intercept = fit_intercept
        self.fit_linear = fit_linear
        self.max_iter = max_iter
        self.max_iter_inner = max_iter_inner
        self.max_rank = max_rank
        self.warm_start = warm_start
        self.tol = tol
        self.refit_iter = refit_iter
        self.eigsh_kwargs = eigsh_kwargs
        self.verbose = verbose
        self.random_state = random_state

    def predict_quadratic(self, X, P=None, lams=None):
        """Prediction from the quadratic term of the factorization machine.

        Returns <Z, XX'>.
        """
        if P is None:
            P = self.P_
            lams = self.lams_

        if not len(lams):
            return 0

        K = polynomial_kernel(X, np.array(P), degree=2, gamma=1, coef0=0)
        return np.dot(K, lams)

    def predict(self, X):
        if self.fit_linear:
            y_hat = self.ridge_.predict(X)
        else:
            y_hat = np.zeros(X.shape[0])
        y_hat += self.predict_quadratic(X)
        return y_hat

    def update_Z(self, X, y, verbose=False, sample_weight=None):
        """Greedy CD solver for the quadratic term of a factorization machine.

        Solves 0.5 ||y - <Z, XX'>||^2_2 + ||Z||_*

        Z implicitly stored as P'ΛP
        """
        n_samples, n_features = X.shape
        rng = check_random_state(self.random_state)
        P = self.P_
        lams = self.lams_
        old_loss = np.inf
        max_rank = self.max_rank
        if max_rank is None:
            max_rank = n_features

        ##
        #residual = self.predict_quadratic(X) - y  # could optimize
        #loss = self._loss(residual, sample_weight=sample_weight)
        #rms = np.sqrt(np.mean((residual) ** 2))
        #print("rank={} loss={}, RMSE={}".format(0, loss, rms))
        ##

        for _ in range(self.max_iter_inner):
            if self.rank_ >= max_rank:
                break
            residual = self.predict_quadratic(X) - y  # could optimize
            if sample_weight is not None:
                residual *= sample_weight
            p = _find_basis(X, residual, **self.eigsh_kwargs)
            P.append(p)
            lams.append(0.)

            # refit
            refit_target = y.copy()
            K = polynomial_kernel(X, np.array(P), degree=2, gamma=1, coef0=0)
            if sample_weight is not None:
                refit_target *= np.sqrt(sample_weight)
                K *= np.sqrt(sample_weight)[:, np.newaxis]
            K = np.asfortranarray(K)
            lams_init = np.array(lams, dtype=np.double)

            # minimizes 0.5 * ||y - K * lams||_2^2 + beta * ||w||_1
            lams, _, _, _ = enet_coordinate_descent(
                lams_init, self.beta, 0, K, refit_target,
                max_iter=self.refit_iter, tol=self.tol, rng=rng, random=0,
                positive=0)
            P = [p for p, lam in zip(P, lams) if np.abs(lam) > 0]
            lams = [lam for lam in lams if np.abs(lam) > 0]
            self.rank_ = len(lams)
            self.quadratic_trace_ = np.sum(np.abs(lams))

            predict_quadratic = self.predict_quadratic(X, P, lams)
            residual = y - predict_quadratic  # y is already shifted
            loss = self._loss(residual, sample_weight=sample_weight)

            if verbose > 0:
                rms = np.sqrt(np.mean((residual) ** 2))
                print("rank={} loss={}, RMSE={}".format(self.rank_, loss, rms))

            if np.abs(old_loss - loss) < self.tol:
                break

            old_loss = loss
        self.P_ = P
        self.lams_ = lams

    def fit(self, X, y, sample_weight=None):
        if not self.warm_start or not hasattr(self, 'P'):
            self.P_ = []
            self.lams_ = []
            self.rank_ = 0

        if sample_weight is not None:
            assert len(sample_weight) == len(y)

        # adjust eigsh defaults
        if 'maxiter' not in self.eigsh_kwargs:
            self.eigsh_kwargs['maxiter'] = 5000
        if 'tol' not in self.eigsh_kwargs:
            self.eigsh_kwargs['tol'] = self.tol


        self.ridge_norm_sq_ = 0
        self.quadratic_trace_ = 0

        if self.fit_linear:
            self.ridge_ = Ridge(alpha=0.5 * self.alpha,
                                fit_intercept=self.fit_intercept)
            old_loss = np.inf
            quadratic_pred = 0

            for i in range(self.max_iter):
                # fit linear
                self.ridge_.fit(X, y - quadratic_pred,
                                sample_weight=sample_weight)
                linear_pred = self.ridge_.predict(X)
                self.ridge_norm_sq_ = np.sum(self.ridge_.coef_ ** 2)

                #print(self._loss(y - (linear_pred + quadratic_pred),
                #                sample_weight=sample_weight))


                # fit quadratic
                self.update_Z(X, y - linear_pred, verbose=self.verbose - 1,
                              sample_weight=sample_weight)
                quadratic_pred = self.predict_quadratic(X)

                loss = self._loss(y - (linear_pred + quadratic_pred),
                                  sample_weight=sample_weight)
                if self.verbose:
                    print("Outer iter {} rank={} loss={}".format(
                        i, self.rank_, loss))
                if np.abs(old_loss - loss) < self.tol:
                    break
                old_loss = loss
        else:
            self.update_Z(X, y, verbose=self.verbose - 1,
                          sample_weight=sample_weight)

        return self

    def _loss(self, residual, sample_weight=None):
        loss = residual ** 2
        if sample_weight is not None:
            loss *= sample_weight
        loss = loss.sum() + self.alpha * self.ridge_norm_sq_
        loss *= 0.5
        loss += self.beta * self.quadratic_trace_
        return loss


def make_multinomial_fm_dataset(n_samples, n_features, rank=5, length=50,
                                random_state=None):
    # Inspired by `sklearn.datasets.make_multilabel_classification`
    rng = check_random_state(random_state)

    X_indices = array.array('i')
    X_indptr = array.array('i', [0])

    for i in range(n_samples):
        # pick a non-zero document length by rejection sampling
        n_words = 0
        while n_words == 0:
            n_words = rng.poisson(length)
        # generate a document of length n_words
        words = rng.randint(n_features, size=n_words)
        X_indices.extend(words)
        X_indptr.append(len(X_indices))

    X_data = np.ones(len(X_indices), dtype=np.float64)
    X = sp.csr_matrix((X_data, X_indices, X_indptr),
                      shape=(n_samples, n_features))
    X.sum_duplicates()

    true_w = rng.randn(n_features)
    true_eigv = rng.randn(rank)
    true_P = rng.randn(rank, n_features)

    y = safe_sparse_dot(X, true_w)
    y += ConvexFM().predict_quadratic(X, true_P, true_eigv)
    return X, y


if __name__ == '__main__':
    n_samples, n_features = 1000, 50
    rank = 5
    length = 5
    X, y = make_multinomial_fm_dataset(n_samples, n_features, rank, length,
                                       random_state=0)
    X, X_val, y, y_val = train_test_split(X, y, test_size=0.25, random_state=0)
    y += 0.01 * np.random.RandomState(0).randn(*y.shape)

    # try ridge
    from sklearn.linear_model import RidgeCV
    ridge = RidgeCV(alphas=np.logspace(-4, 4, num=9, base=10),
                    fit_intercept=False)
    ridge.fit(X, y)
    y_val_pred = ridge.predict(X_val)
    print('RidgeCV validation RMSE={}'.format(
        np.sqrt(mean_squared_error(y_val, y_val_pred))))

    # convex factorization machine path
    fm = ConvexFM(fit_linear=True, warm_start=False, max_iter=20, tol=1e-4,
                  max_iter_inner=50, fit_intercept=True,
                  eigsh_kwargs={'tol': 0.1})

    if True:
        for alpha in (0.01, 0.1, 1, 10):
            for beta in (10000, 500, 150, 100, 50, 1, 0.001):
                fm.set_params(alpha=alpha, beta=beta)
                fm.fit(X, y)
                y_val_pred = fm.predict(X_val)
                print("α={} β={}, rank={}, validation RMSE={:.2f}".format(
                    alpha,
                    beta,
                    fm.rank_,
                    np.sqrt(mean_squared_error(y_val, y_val_pred))))

    if False:
        fm.set_params(alpha=0.1, beta=1, fit_linear=True)

        import scipy.sparse as sp

        fm.fit(sp.vstack([X, X]), np.concatenate([y, y]))
        y_val_pred = fm.predict(X_val)
        print("FM rank={}, validation RMSE={:.2f}".format(
            fm.rank_,
            np.sqrt(mean_squared_error(y_val, y_val_pred))))

        fm.set_params(beta=1.)
        fm.fit(X, y, sample_weight=2 * np.ones_like(y))
        y_val_pred = fm.predict(X_val)
        print("FM rank={}, validation RMSE={:.2f}".format(
            fm.rank_,
            np.sqrt(mean_squared_error(y_val, y_val_pred))))
