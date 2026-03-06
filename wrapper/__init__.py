__all__ = ["RWKVTokenizer", "ModRWKVProcessor"]


def __getattr__(name):
    if name == "RWKVTokenizer":
        from .tokenizer import RwkvTokenizer

        return RwkvTokenizer
    if name == "ModRWKVProcessor":
        from .processor import ModRWKVProcessor

        return ModRWKVProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
