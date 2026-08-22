"""PreCURE campaign-expression optimization."""

from .models import Campaign, Evaluation, Expression, Persona
from .objective import campaign_proposal_score, calibrated_target, quality_term

__all__ = [
    "Campaign",
    "Evaluation",
    "Expression",
    "Persona",
    "campaign_proposal_score",
    "calibrated_target",
    "quality_term",
]

__version__ = "0.1.0"
