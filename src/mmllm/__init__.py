"""mosaic-qwen: Qwen3.5 / Qwen3.8 with every softmax-attention layer replaced by an official
Memory Mosaics block (mmllm.official loads the unmodified upstream code from
third_party/MemoryMosaics).

    from mmllm import MQConfig, convert_to_mosaic_qwen, load_mosaic_qwen, load_standalone
"""
_MAIN = ("MQConfig", "MosaicQwen", "MosaicQwenLayer", "convert_to_mosaic_qwen", "load_mosaic_qwen", "load_standalone")
__all__ = list(_MAIN)


def __getattr__(name):          # imported on first use (it loads the upstream package)
    if name in _MAIN:
        from . import mosaic_qwen
        return getattr(mosaic_qwen, name)
    raise AttributeError(name)
