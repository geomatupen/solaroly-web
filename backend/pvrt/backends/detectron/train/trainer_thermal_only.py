"""
Trainer (thermal-only) stub
----------------------------

Historically this module provided a Detectron2 trainer for true
single-channel thermal-only training. The project no longer
supports single-channel models — we coerce any '1' requests to 3
(thermal encoded as 3-channel grayscale) and use RGB+thermal (4ch)
when thermal is present and explicitly requested.

This file remains as a lightweight stub to avoid import-time breakage
in code that may still reference the old symbol. If you still import
ThermalOnlyTrainer, importing this module will raise a clear error.
"""

def ThermalOnlyTrainer(*args, **kwargs):
    raise NotImplementedError("Thermal-only trainer has been removed. Use RGB+Thermal (4ch) or thermal-as-RGB (3ch) workflows.")
