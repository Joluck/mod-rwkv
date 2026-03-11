__all__ = [
    "RWKVTokenizer",
    "ModRWKVProcessor",
    "ModRWKVConfig",
    "ModRWKVProjectorConfig",
    "RWKV7VLModel",
    "RWKV7VLForConditionalGeneration",
]


def __getattr__(name):
    if name == "RWKVTokenizer":
        from .tokenizer import RwkvTokenizer
        return RwkvTokenizer
    if name == "ModRWKVProcessor":
        from .processor import ModRWKVProcessor
        return ModRWKVProcessor
    if name == "ModRWKVConfig":
        from .modeling_modrwkv import ModRWKVConfig
        return ModRWKVConfig
    if name == "ModRWKVProjectorConfig":
        from .modeling_modrwkv import ModRWKVProjectorConfig
        return ModRWKVProjectorConfig
    if name == "RWKV7VLModel":
        from .modeling_modrwkv import RWKV7VLModel
        return RWKV7VLModel
    if name == "RWKV7VLForConditionalGeneration":
        from .modeling_modrwkv import RWKV7VLForConditionalGeneration
        return RWKV7VLForConditionalGeneration
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
