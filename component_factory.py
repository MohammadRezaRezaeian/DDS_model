# Concrete class implementations (simulated targets from your OOP directories)
from model.bayesian_tensor import BayesianTensorDDS
from metric.mse_loss import MSELoss
from metric.CPFrobeniusLoss import CPFrobeniusLoss
from optimizer.kalman_momentum import KalmanMomentum
from optimizer.CPSGDOptimizer import CPSGDOptimizer

class ComponentFactory:
    """Decouples component generation logic from the primary engine state loop."""
    
    @staticmethod
    def create_model(model_name: str, metric, optimizer, params: dict) -> object:
        registry = {
            "BayesianTensor": lambda: BayesianTensorDDS(
                metric=metric,
                optimizer=optimizer,
                params=params
            )
        }
        if model_name not in registry:
            raise ValueError(f"Unknown model type: {model_name}")
        return registry[model_name]()

    @staticmethod
    def create_metric(metric_name: str) -> object:
        registry = {
            "MSE": MSELoss,
            "CPFrobenius": CPFrobeniusLoss
        }
        if metric_name not in registry:
            raise ValueError(f"Unknown metric type: {metric_name}")
        return registry[metric_name]()

    @staticmethod
    def create_optimizer(optimizer_name: str) -> object:
        registry = {
            "KalmanMomentum": KalmanMomentum,
            "CPSGDOptimizer": CPSGDOptimizer
        }
        if optimizer_name not in registry:
            raise ValueError(f"Unknown optimizer type: {optimizer_name}")
        return registry[optimizer_name]()