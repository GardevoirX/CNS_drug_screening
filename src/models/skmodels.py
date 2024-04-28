from sklearn.linear_model import LogisticRegression
from ._abc import ABC

class LogisticRegressionModel(SkLearnModelsABC):
    def __init__(self, penalty='l2', tol=0.0001, C=1.0, fit_intercept=True,
                 class_weight=None, solver='lbfgs', max_iter=100, verbose=0,
                 n_jobs=None, l1_ratio=None):
        self.model = LogisticRegression(penalty=penalty, tol=tol, C=C, fit_intercept=fit_intercept,
                                        class_weight=class_weight, solver=solver, max_iter=max_iter, verbose=verbose,
                                        n_jobs=n_jobs, l1_ratio=l1_ratio)
