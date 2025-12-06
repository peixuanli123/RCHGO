# RCHGO

<p align="justify">
RCHGO is a deep learning method for protein function prediction that takes an amino acid sequence as input and outputs GO terms with confidence scores. This model consists of two modules, i.e., (A) manually crafted feature-based GO prediction (MCFGP) and (B) PLM-based GO prediction (PLMGP), which are ensembled at the decision level via a fully connected network. 
</p>

<p align="justify">
MCFGP derives three manually crafted features from protein sequences across different biological perspectives, together with a predicted contact map, which are then processed by a residual graph convolutional network. Meanwhile, a protein-level cross-attention module is employed to further refine these representations by modeling the interactions between protein features and GO semantics. Manually crafted feature representations include position-specific scoring matrix (PSSM), hybrid structure encoding matrix (HSCM), and family-domain binary indicator vector (FDBIV). PLMGP leverages feature representations obtained from the pre-trained ProtT5-XL-UniRef50 model and feeds them into an RGCN equipped with a residue-level cross-attention module to learn fine-grained associations between residues and GO terms. The outputs of the MCFGP and PLMGP are fused at the decision level via a fully connected network to produce the final RCHGO predictions. 
</p>

## System Requirements
### 1. Conda Environment: 
<li> python==3.11.5  </li>
  
<li> tensorflow-gpu==2.7.2 </li>  
  
<li> pytorch==2.1.0+cu121  </li>
   
<li> CUDA>=12.1  </li>
   
<li> cudnn>=8.9.7 </li>  

### 2. Software  
<li> <a href="https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/">BLAST</a> &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp###### &nbsp generate PSSM feature &nbsp ###### </li>
<li> <a href="https://www.ebi.ac.uk/interpro/download/">InterProScan</a>&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp###### &nbsp generate FDBIV feature &nbsp ######  </li>
<li> <a href="https://github.com/google-deepmind/alphafold">AlphaFold2</a>&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp###### &nbsppredict 3D structures of proteins &nbsp ######  </li>
<li> <a href="https://github.com/agemagician/ProtTrans">ProtT5-XL-UniRef50</a> &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp###### &nbsp generate protein language model-based feature &nbsp ######  </li>


### 3. Data
<li> <a href="https://www.uniprot.org/help/downloads">Swiss-Prot database</a>  </li>

### 4. MCFGP
4.1 Generate PSSM feature  
    python PSSM.py ./test_sequence.fasta ./sequence/ ./original_pssm/ ./log_pssm/ 30  
4.2 Predict 3D structures  
    sh run_alphafold.sh --fasta_paths ./test_sequence.fasta --output_dir ./alphafold_predictions/   
4.3 








