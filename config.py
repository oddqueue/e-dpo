import warnings
from dataclasses import dataclass, field

from trl import DPOConfig


@dataclass
class EpsilonDPOConfig(DPOConfig):
    epsilon: float = field(
        default=0.01,
        metadata={
            "help": "Parameter controlling the step size of KL penalty relaxation."
        },
    )

    def __post_init__(self):
        if self.reference_free == True:
            warnings.warn(
                "When using `EpsilonDPOTrainer`, you should set `reference_free=False`. "
                "We have set it for you, but you should do it yourself in the future."
            )
            self.reference_free = False

        if self.precompute_ref_log_probs == True:
            warnings.warn(
                "When using `EpsilonDPOTrainer`, you should set `precompute_ref_log_probs=True`. "
                "We have set it for you, but you should do it yourself in the future."
            )
            self.precompute_ref_log_probs = False

        if self.loss_type != "sigmoid":
            warnings.warn(
                "When using `EpsilonDPOTrainer`, you should set `loss_type=\"sigmoid\"`. "
                "We have set it for you, but you should do it yourself in the future."
            )           
            self.loss_type = "sigmoid" 

        super().__post_init__()