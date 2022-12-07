import sklearn.metrics as sm
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from collections.abc import Iterable, Callable
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod


class CostFunction(ABC):
    """Abstract class for cost functions"""

    def __init__(self, metrics: Iterable[str], M: "np.ndarray[float]") -> None:
        """Cost function constructor.

        Args:
            metrics (Iterable[str]): Iterable of strings of the form (metric_name).
            M (np.ndarray[float]): Positive definite matrix of size len(metrics).

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        self.metrics = metrics
        self.M = M or np.identity(len(metrics))  # type: ignore
        self._check_positive_definite(self.M)

    @abstractmethod
    def objective(
        self, y_true: "np.ndarray[float]", y_pred: "np.ndarray[float]"
    ) -> float:
        """Objective function.

        Args:
            y_true (np.ndarray[float]): Array-like of true labels of length N.
            y_pred (np.ndarray[float]): Array-like of predicted labels of length N.
        """
        pass

    @staticmethod
    def _to_array(y: Iterable[float]) -> "np.ndarray[float]":
        return np.fromiter(y, float)

    @staticmethod
    def _check_positive_definite(M: "np.ndarray[float]") -> None:
        if not np.all(np.linalg.eigvals(M) > 0):
            raise ValueError(f"Matrix {M} is not positive definite")

    def make_scorer(self) -> Callable:
        return sm.make_scorer(self.objective, greater_is_better=False)

    def __call__(self, y_true: Iterable[float], y_pred: Iterable[float]) -> float:
        y_pred_array = self._to_array(y_pred)
        y_true_array = self._to_array(y_true)

        return self.objective(y_true_array, y_pred_array)


class ClassificationCostFunction(CostFunction):
    def __init__(
        self,
        metrics: Iterable[str],
        M: "np.ndarray[float]" = None,
        metric_class_opt_val_map: dict[str, tuple[str, float]] = None,
        proba_threshold: float = 0.5,
    ):
        """Defines cost functional for optimization of multiple metrics.
        Since this is defined as a loss function, cross validation returns the negative of the score [1].

        Args:
            metrics (Iterable[str]): Iterable of strings of the form (metric_name).
            M (np.ndarray[float]): Positive definite matrix of size len(metrics).
            metric_class_map (dict[str, str], optional): Dictionary mapping metric to class or probability of the form {'metric': 'class' or 'proba'}. Defaults to {}.
            proba_threshold (float, optional): Probability threshold used to convert probabilities into classes. Defaults to 0.5.

        References:
            [1] https://github.com/scikit-learn/scikit-learn/issues/2439

        Example:
            >>> y_true = [0, 0, 0, 1, 1]
            >>> y_pred = [0.46, 0.6, 0.29, 0.25, 0.012]
            >>> threshold = 0.5
            >>> metrics = ["f1_score", "roc_auc_score"]
            >>> cf = ClassificationCostFunction(metrics)
            >>> np.isclose(cf(y_true, y_pred), 1.41, rtol=1e-01, atol=1e-01)
            True
            >>> X, y = make_classification()
            >>> model = LogisticRegression()
            >>> model.fit(X, y)
            >>> y_proba = model.predict_proba(X)[:, 1]
            >>> cost = cf(y, y_proba)
            >>> f1 = getattr(sm, "f1_score")
            >>> roc_auc = getattr(sm, "roc_auc_score")
            >>> y_pred = np.where(y_proba > 0.5, 1, 0)
            >>> scorer_output = np.sqrt((f1(y, y_pred) - 1.0)**2 + (roc_auc(y, y_proba) - 1.0)**2)
            >>> np.isclose(cost, scorer_output)
            True
        """
        super().__init__(metrics, M)
        self.proba_threshold = proba_threshold
        self.metric_class_opt_val_map = metric_class_opt_val_map or {
            "accuracy_score": ("class", 1),
            "f1_score": ("class", 1),
            "log_loss": ("class", 0),
            "precision_score": ("class", 1),
            "recall_score": ("class", 1),
            "roc_auc_score": ("proba", 1),
        }

    def _to_class(self, array: "np.ndarray[float]", metric: str) -> "np.ndarray[float]":
        """Convert probability to class.

        Args:
            array (np.ndarray[float]): Array of probabilities of size (n_samples, 1).
            metric (str): Metric that requires class.

        Returns:
            np.ndarray[float]: Converted array of size (n_samples, 1).
        """
        # sourcery skip: inline-immediately-returned-variable
        output = (
            np.where(array > self.proba_threshold, 1, 0)
            if self.metric_class_opt_val_map[metric][0] == "class"
            else array
        )

        return output

    def objective(
        self, y_true: "np.ndarray[float]", y_pred: "np.ndarray[float]"
    ) -> float:

        self._check_positive_definite(self.M)

        opt_values = np.array(
            [self.metric_class_opt_val_map[metric][1] for metric in self.metrics]
        )

        metric_values = np.array(
            [
                getattr(sm, metric)(y_true, self._to_class(y_pred, metric))
                for metric in self.metrics
            ]
        )

        return np.sqrt(
            np.dot(
                np.dot(metric_values - opt_values, self.M), metric_values - opt_values
            )
        )
