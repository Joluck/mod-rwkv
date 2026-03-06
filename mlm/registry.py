from typing import Dict, Any, Type
from .projector.modules import VisualAdapter
import torch.nn as nn
from .encoder.speech_encoder import SpeechEncoder
from .encoder.whisper_encoder import WhisperEncoder
from .encoder.clip_encoder import ClipEncoder
from .encoder.siglip_encoder import SiglipEncoder

Projector_Registry: Dict[str, Type[nn.Module]] = {
    "siglip2": VisualAdapter,
    # "simple": SimpleProjection,
    # "mlp":    MLPAdapter,
}

Encoder_Registry: Dict[str, Type[nn.Module]] = {
    "clip": ClipEncoder,
    "whisper": WhisperEncoder,
    "speech": SpeechEncoder,
    "siglip2": SiglipEncoder,
}

