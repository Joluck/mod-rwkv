from typing import Dict, Any, Type
from .projector.modules import VisualAdapter, ModalityProjector, VlProj
import torch.nn as nn
from .encoder.speech_encoder import SpeechEncoder
from .encoder.whisper_encoder import WhisperEncoder
from .encoder.clip_encoder import ClipEncoder
from .encoder.siglip_encoder import SiglipEncoder
from .encoder.siglip2 import Siglip2Encoder
from .encoder.qwen3_5 import Qwen3_5VLEncoder
Projector_Registry: Dict[str, Type[nn.Module]] = {
    "siglip": VisualAdapter,
    "siglip2": ModalityProjector,
    "auto_siglip2": VlProj,
    "qwen3_5": VisualAdapter,
    # "simple": SimpleProjection,
    # "mlp":    MLPAdapter,
}

Encoder_Registry: Dict[str, Type[nn.Module]] = {
    "clip": ClipEncoder,
    "whisper": WhisperEncoder,
    "speech": SpeechEncoder,
    "siglip": SiglipEncoder,
    "siglip2": Siglip2Encoder,
    "auto_siglip2": Siglip2Encoder,
    "qwen3_5": Qwen3_5VLEncoder
}

