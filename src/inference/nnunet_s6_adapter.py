"""Install process-local scheme 6 hooks without changing nnUNet or its plans."""
from functools import wraps

from nnunet_s6_preprocess import ConfigurationProxy
from nnunet_s6_merge import export_prediction_from_logits


def install_scheme6(prediction):
    """Configure the predictor before it sends a picklable proxy to workers."""
    predictor = prediction.nnUNetPredictor
    original = predictor.initialize_from_trained_model_folder
    if not getattr(original, '_autoprintfilm_scheme6', False):
        @wraps(original)
        def initialize(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            self.configuration_manager = ConfigurationProxy(self.configuration_manager)
            return result

        initialize._autoprintfilm_scheme6 = True
        predictor.initialize_from_trained_model_folder = initialize
    prediction.export_prediction_from_logits = export_prediction_from_logits
