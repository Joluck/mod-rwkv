from typing import Dict, Any, Type
from .projector.modules import VisualAdapter, ModalityProjector, VlProj
import torch.nn as nn

from .encoder.qwen3_5 import Qwen3_5VLEncoder
Projector_Registry: Dict[str, Type[nn.Module]] = {
    "qwen3_5": VisualAdapter,
    # "simple": SimpleProjection,
    # "mlp":    MLPAdapter,
}

Encoder_Registry: Dict[str, Type[nn.Module]] = {
    "qwen3_5": Qwen3_5VLEncoder
}

