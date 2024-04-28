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
        max_iter=100,
        l1_ratio=None,
    ):
        self.model = LogisticRegression(
            penalty=penalty,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            class_weight=class_weight,
            solver=solver,
            max_iter=max_iter,
            l1_ratio=l1_ratio,
        )


class LinearRegressionModel(SKLearnModelsABC):
    def __init__(self, fit_intercept=True):
        self.model = LinearRegression(fit_intercept=fit_intercept)


class RidgeRegressionModel(SKLearnModelsABC):
    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=None, tol=0.001):
        self.model = Ridge(
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
        )


class LassoRegressionModel(SKLearnModelsABC):
    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=None, tol=0.001):
        self.model = Lasso(
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
        )


class ElasticNetRegressionModel(SKLearnModelsABC):
    def __init__(
        self,
        alpha=1.0,
        l1_ratio=0.5,
        fit_intercept=True,
        max_iter=None,
        tol=0.001,
    ):
        self.model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
        )


class BayesianRidgeRegressionModel(SKLearnModelsABC):
    def __init__(self, n_iter=300, tol=0.001):
        self.model = BayesianRidge(n_iter=n_iter, tol=tol)


class SGDClassifierModel(SKLearnModelsABC):
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
        epsilon=0.1,
        n_jobs=None,
        random_state=None,
        learning_rate="optimal",
        eta0=0.0,
        power_t=0.5,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        class_weight=None,
        warm_start=False,
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
            epsilon=epsilon,
            n_jobs=n_jobs,
            random_state=random_state,
            learning_rate=learning_rate,
            eta0=eta0,
            power_t=power_t,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            class_weight=class_weight,
            warm_start=warm_start,
        )


class KernelRidgeModel(SKLearnModelsABC):
    def __init__(self, alpha=1.0, kernel="linear", degree=3, gamma=None, coef0=1.0):
        self.model = KernelRidge(
            alpha=alpha, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0
        )


class SVCModel(SKLearnModelsABC):
    def __init__(
        self,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        shrinking=True,
        probability=False,
        tol=0.001,
        cache_size=200,
        class_weight=None,
        verbose=False,
        max_iter=-1,
        decision_function_shape="ovr",
        break_ties=False,
        random_state=None,
    ):
        self.model = SVC(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state,
        )


class KNNModel(SKLearnModelsABC):
    def __init__(
        self,
        n_neighbors=5,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
    ):
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )


class KmeansModel(SKLearnModelsABC):
    def __init__(
        self, n_clusters=8, init="k-means++", max_iter=300, n_init=10, random_state=None
    ):
        self.model = KMeans(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
        )


class GaussianmixtureModel(SKLearnModelsABC):
    def __init__(
        self,
        n_components=1,
        covariance_type="full",
        tol=0.001,
        reg_covar=1e-06,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
    ):
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            random_state=random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval,
        )
