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
<li> <a href="https://github.com/PDB-REDO/dssp">DSSP</a>&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp###### &nbsp generate HSCM feature  &nbsp ######  </li>
<li> <a href="https://github.com/agemagician/ProtTrans">ProtT5-XL-UniRef50</a> &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp###### &nbsp generate protein language model-based feature &nbsp ######  </li>


### 3. Data
<li> <a href="https://www.uniprot.org/help/downloads">Swiss-Prot database</a>  </li>
<li> <a href="https://www.ebi.ac.uk/GOA/index"</a>  Gene Ontology annotation database </li>
<li> <a href="https://alphafold.ebi.ac.uk"</a>  Alphafold database </li>
<li> <a href="https://alphafold.ebi.ac.uk"</a>  Benchmark dataset </li>
<li> <a href="https://alphafold.ebi.ac.uk"</a>  Library </li>

### 4. MCFGP
<b>4.1 Generate PSSM feature</b>  
    python ./MCFGP/PSSM.py argv[1] argv[2] argv[3] argv[4] argv[5]   
    argv[1]: sequence file with fasta format (input1)  
    argv[2]: sequence directory (output1)   
    argv[3]: original pssm directory (output2)   
    argv[4]: logistic pssm directory (output3, **** we need ****)   
    argv[5]: thread number in multiple threading (input2)  
      
    e.g., python ./MCFGP/PSSM.py ./test_sequence.fasta ./sequence/ ./original_pssm/ ./test_pssm/ 30  
<b>4.2 Predict 3D structures</b>  
    sh run_alphafold.sh -d argv[1] -o argv[2] -f argv[3] -t argv[4] --gpu_device=0   
    argv[1]: Alphafold2 database directory (input1)     
    argv[2]: predicted structure directory (output1, **** we need ****)   
    argv[3]: sequence file with fasta format (input2)   
    argv[4]: database version (input3)
      
    e.g., sh run_alphafold.sh -d /data/library/database -o ./test_structures/ -f ./test_sequence.fasta -t 2020-05-14 --gpu_device=0 

<b>4.3 Generate HSCM feature</b>  
    python Create_Predict_Structure_Feature.py argv[1] argv[2]  
    argv[1]: predicted structures directory (input1)       
    argv[2]: HSCM feature directory (output1, **** we need ****)     
    
    e.g., python Create_Predict_Structure_Feature.py /test_structures/  ./test_hscm/  

<b>4.4 Generate FDBIV feature</b>  
    (1) python run_interproscan.py argv[1] argv[2] argv[3]  
    argv[1]: workdir (input1)  
    argv[2]: split number of test sequences (input2)    
    argv[3]: thread number in multiple threading (input3)  

    e.g., python run_interproscan.py ./interpro/ 100 30
    
(2) python postdeal_interproscan.py argv[1] argv[2] argv[3]    
    argv[1]: interpro result directory (input1)  
    argv[2]: splited interpro result directory (output1)  
    argv[3]: splited interpro array directory (output2, **** we need ****)

    e.g., python postdeal_interproscan.py ./interpro/results/ ./interpro/entry_name/ ./interpro/entry_array/

<b>4.5 Create labels for training, validation, and test datasets</b>  
   python Create_Label_Benchmark.py argv[1]  
   argv[1]: workdir (input1)    

    e.g., python Create_Label_Benchmark.py ./data/


<b>4.6 Create graph from predicted structures</b>  
   (1) python generate_points.py argv[1] argv[2]    
   argv[1]: workdir (input1)  
   argv[2]: go type (input2)

    e.g., python generate_points.py ./data/ mf/bp/cc
   (2) python process_graph_mcfgp.py -d argv[1] -t argv[2]  
   argv[1]: go type  (input1)  
   argv[2]: threshold for determining contact map (input2)

    e.g., python process_graph_mcfgp.py -d mf/bp/cc -t 10 

  (3) python Create_InterPro_Array.py  argv[1] argv[2]  
  argv[1]:  workdir (input1)   
  argv[2]: go type (input2)  

    e.g., Create_InterPro_Array.py ./data/ mf/bp/cc 
   
<b>4.7 Train and evaluate MCFGP</b>    
   python Run_MCFGP.py argv[1] argv[2]  
   argv[1]: go type (input1)  
   argv[2]: gpu id (input2)   

    e.g., python Run_MCFGP.py mf/bp/cc 0
   
### 5. PLMGP
<b>5.1 Generate PLM embeddings</b>    
   python prottrans_extract.py -i argv[1] -o argv[2]   
   argv[1]: sequence file with fasta format (input1)    
   argv[2]: feature embedding directory (output1, **** we need ****)  

    e.g., python prottrans_extract.py -i ./test_sequence.fasta -o ./plm_embeddings/

<b>5.2 Create labels for training, validation, and test datasets</b>  
   python Create_Label_Benchmark.py argv[1]  
   argv[1]: workdir (input1)    

    e.g., python Create_Label_Benchmark.py ./data/

<b>5.3 Create graph from predicted structures</b>  
   (1) python generate_points.py argv[1] argv[2]    
   argv[1]: workdir (input1)  
   argv[2]: go type (input2)

    e.g., python generate_points.py ./data/ mf/bp/cc
   (2) python process_graph_plmgp.py -d argv[1] -t argv[2]  
   argv[1]: go type  (input1)  
   argv[2]: threshold for determining contact map (input2)

    e.g., python process_graph_plmgp.py -d mf/bp/cc -t 10 

<b>5.4 Train and evaluate PLMGP</b>    
   python Run_PLMGP.py argv[1] argv[2]  
   argv[1]: go type (input1)  
   argv[2]: gpu id (input2)   

    e.g., python Run_PLMGP.py mf/bp/cc 0

### 6. RCHGO
<b>6.1 Copy MCFGP/PLMGP prediction results to workspace</b>  
python Copy_Results_LLM.py argv[1] argv[2]  
argv[1]: original prediction result directory (input1)  
argv[2]: copy prediction result directory (output1)  

    e.g., python Copy_Results_LLM.py ./mcfgp_results/ ./ensemble/  
    e.g., python Copy_Results_LLM.py ./plmgp_results/ ./ensemble/









