# RCHGO

RCHGO is a deep learning method for protein function prediction that takes an amino acid sequence as input and outputs GO terms with confidence scores. This model consists of two modules, i.e., (A) manually crafted feature-based GO prediction (MCFGP) and (B) PLM-based GO prediction (PLMGP), which are ensembled at the decision level via a fully connected network. 

MCFGP derives three manually crafted features from protein sequences across different biological perspectives, together with a predicted contact map, which are then processed by a residual graph convolutional network. Manually crafted feature representations include position-specific scoring matrix (PSSM), hybrid structure encoding matrix (HSCM), and family-domain binary indicator vector (FDBIV).

