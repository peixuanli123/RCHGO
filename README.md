# RCHGO

RCHGO is a deep learning method for protein function prediction that takes an amino acid sequence as input and outputs GO terms with confidence scores. This model consists of two modules, i.e., (A) manually crafted feature-based GO prediction (MCFGP) and (B) PLM-based GO prediction (PLMGP), which are ensembled at the decision level via a fully connected network. 

MCFGP derives three manually crafted features from protein sequences across different biological perspectives, together with a predicted contact map, which are then processed by a residual graph convolutional network. Meanwhile, a protein-level cross-attention module is employed to further refine these representations by modeling the interactions between protein features and GO semantics. Manually crafted feature representations include position-specific scoring matrix (PSSM), hybrid structure encoding matrix (HSCM), and family-domain binary indicator vector (FDBIV).

PLMGP leverages feature representations obtained from the pre-trained ProtT5-XL-UniRef50 model and feeds them into an RGCN equipped with a residue-level cross-attention module to learn fine-grained associations between residues and GO terms. 

The outputs of the MCFGP and PLMGP are fused at the decision level via a fully connected network to produce the final RCHGO predictions. 







