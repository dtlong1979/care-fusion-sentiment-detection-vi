"""CARE-Fusion: regime-aware fusion of text and affective markers for
Vietnamese emotion classification (ViGoEmotions, 6 groups).

Package layout follows the experimental protocol (docs/):
  preprocess  -> Part A
  resources   -> Part B (q_j, weak regime labels, PMI graph)  [built from TRAIN only]
  model       -> Part C
  losses      -> Part D1
  train       -> Part D2-D4
  baselines   -> Part E
  evaluate    -> Part F-G
"""

__version__ = "0.1.0"
