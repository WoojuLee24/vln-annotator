"""
vln_annotator — VLN Dataset Auto-Annotator
==========================================
Generates GT-compatible R2R navigation instructions from path geometry + rendered images.
"""
from .config import AnnotatorConfig
from .dataset_assembler import VLNTokenizer, assemble_dataset, save_dataset
from .instruction_gen import compute_stats, print_stats
from .path_analysis import analyze_path, primitives_to_text
from .pipeline import run

__version__ = "1.0.0"
__all__ = [
    "AnnotatorConfig",
    "VLNTokenizer",
    "assemble_dataset",
    "save_dataset",
    "compute_stats",
    "print_stats",
    "analyze_path",
    "primitives_to_text",
    "run",
]
