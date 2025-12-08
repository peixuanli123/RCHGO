import os
import pickle as pkl
import sys

import click
from Bio import SeqIO
from tqdm import tqdm

def get_pid_list(fasta_file):
    pid_list = []
    for seq in SeqIO.parse(fasta_file, 'fasta'):
        pid_list.append(seq.id)

    return pid_list

def read_sequence(sequence_file):

    f = open(sequence_file, "r")
    text = f.read()
    f.close()

    sequence_dict = dict()
    for line in text.splitlines():
        line = line.strip()
        if(line.startswith(">")):
            name = line
        else:
            sequence_dict[name[1:]] = line

    return sequence_dict

def read_pdb(pdb_file, protein_length):
    #construct pdb ca points
    points = []
    with open(pdb_file, 'r') as f:
        for line in f:
            
            if line.startswith('ATOM'):
                point_type = line[12:16].strip() #col 13-16
                if point_type == 'CA':
                    x = float(line[30:38].strip()) #col 31-38
                    y = float(line[38:46].strip()) #col 39-46
                    z = float(line[46:54].strip()) #col 47-54
                    amino = line[17:20].strip() #col 18-20
                    points.append((x, y, z, amino))
    if len(points)>0:
        try:
            assert protein_length==len(points)
        except:
            return False
    return points

def main(pid_list_file, output_file, type, data_type):
    with open(pid_list_file, 'rb') as fr:
        pid_list = pkl.load(fr)

    sequence_dict = read_sequence("/data/yihengzhu/MKFGO_2.0/resource/benchmark/all_dataset_sequence.fasta")
    
    pdb_points_info = {}
    for protein in tqdm(pid_list):
        pdb_file = './data/PDB_folder/{0}/{1}/{2}.pdb'.format(type, data_type, protein)
        if os.path.exists(pdb_file):
            acid_points = read_pdb(pdb_file, len(sequence_dict[protein]))
            if acid_points==False:
                print("Wrong sequence length!!!")
                return False
            elif len(acid_points)==0:
                print("Empty PDB file!!!")
                return False
            else:
                pdb_points_info[protein] = acid_points
        else:
            print("Unseen proteins!!!")
            return False
    with open('./data/{}.pkl'.format(output_file), 'wb') as fw:
        pkl.dump(pdb_points_info, fw)
        print("Result save as: ./data/{}.pkl".format(output_file))


if __name__ == '__main__':

    workdir = sys.argv[1]
    type = sys.argv[2]
    data_type_list = ["train", "evaluate", "test"]

    for data_type in data_type_list:
        main(workdir + "/" + type + "_" + data_type + "_used_pid_list.pkl", type + "_" + data_type + "_pdb_points", type, data_type)
