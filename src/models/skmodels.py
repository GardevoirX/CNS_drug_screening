from sklearn.cluster import KMeans
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    BayesianRidge,
    SGDClassifier,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from ._abc import SKLearnModelsABC


class LogisticRegressionModel(SKLearnModelsABC):
    def __init__(
        self,
        penalty="l2",
        tol=0.0001,
        C=1.0,
        fit_intercept=True,
        class_weight=None,
        solver="lbfgs",
        max_iter=1000,
        l1_ratio=None
    ):
        self.model = LogisticRegression(
            penalty=penalty,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            class_weight=class_weight,
            solver=solver,
            max_iter=max_iter,
            l1_ratio=l1_ratio
        )


class LinearRegressionModel(SKLearnModelsABC): #least squares
    def __init__(self, fit_intercept=True):
        self.model = LinearRegression(fit_intercept=fit_intercept)


class RidgeRegressionModel(SKLearnModelsABC): #ridge regression
    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=None, tol=0.001):
        self.model = Ridge(
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol
        )


class LassoRegressionModel(SKLearnModelsABC): #lasso regression
    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=1000, tol=0.001):
        self.model = Lasso(
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol)   

class ElasticNetRegressionModel(SKLearnModelsABC):  #elastic-net
    def __init__(
        self,
        alpha=1.0,
        l1_ratio=0.5,
        fit_intercept=True,
        max_iter=1000,
        tol=0.001
    ):
        self.model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol
        )


class BayesianRidgeRegressionModel(SKLearnModelsABC):  #bayesian regression
    def __init__(self, max_iter=300, tol=0.001):
        self.model = BayesianRidge(max_iter=max_iter, tol=tol)


class SGDClassifierModel(SKLearnModelsABC):  #sgd classifier
    def __init__(
        self,
        loss="hinge",
        penalty="l2",
        alpha=0.0001,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=1000,
        tol=0.001,
        shuffle=True,
        verbose=0,
        epsilon=0.1
    ):
        self.model = SGDClassifier(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            shuffle=shuffle,
            verbose=verbose,
            epsilon=epsilon
        )


class KernelRidgeModel(SKLearnModelsABC):  #kernel ridge
    def __init__(self, alpha=1.0, kernel="linear", degree=3, gamma=None, coef0=1.0):
        self.model = KernelRidge(
            alpha=alpha, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0
        )


class SVCModel(SKLearnModelsABC):  #SVM
    def __init__(
        self,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        shrinking=True,
        probability=False,
        tol=0.001
    ):
        self.model = SVC(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol
        )


class KNNModel(SKLearnModelsABC):  #KNN
    def __init__(
        self,
        n_neighbors=5,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2
    ):
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p
        )


class KmeansModel(SKLearnModelsABC): #K-means
    def __init__(
        self, n_clusters=2, init="k-means++", max_iter=300, n_init=10, random_state=None
    ):
        self.model = KMeans(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
        )


class GaussianmixtureModel(SKLearnModelsABC): #GMM
    def __init__(
        self,
        n_components=1,
        covariance_type="full",
        tol=0.001,
        reg_covar=1e-06,
        max_iter=100,
        n_init=1,
    ):
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init
        )
