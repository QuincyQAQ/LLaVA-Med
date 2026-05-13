from llava.decoding.registry import register_decoding
from llava.decoding.strategies.vgs.generate import generate_llava_med_vgs

register_decoding("vgs", generate_llava_med_vgs)

__all__ = ["generate_llava_med_vgs"]
