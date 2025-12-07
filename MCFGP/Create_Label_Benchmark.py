import os
import pickle as pkl
import sys

backdir = "/data/yihengzhu/MKFGO_2.0/resource/benchmark/experiments/back/"
pdb_dir = "/data/yihengzhu/MKFGO_2.0/resource/benchmark/alphafold_structures/"
all_sequence_file = "/data/yihengzhu/MKFGO_2.0/resource/benchmark/all_dataset_sequence.fasta"



def read_protein_list(protein_list_file):

    f = open(protein_list_file, "r")
    text = f.read()
    f.close()

    return text.splitlines()

def read_sequence_dict(sequence_file):

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


def read_label(labelfile):  # read label

    label_dict = dict()
    f=open(labelfile,"r")
    line_txt = f.read()
    f.close()


    for line in line_txt.splitlines():

        values = line.strip().split()
        term_list = values[1].split(",")
        label_dict[values[0]] = term_list

    return label_dict

def create_label(workdir):

    type_list = ["MF", "BP", "CC"]
    data_type_list = ["train", "evaluate", "test"]
    sequence_dict = read_sequence_dict(all_sequence_file)

    for data_type in data_type_list:

        protein_list_file = workdir + "/" + data_type + "_protein_list"
        protein_list = read_protein_list(protein_list_file)

        for type in type_list:

            label_file = backdir + "/" + type + "/" + data_type + "_gene_label"
            label_dict = read_label(label_file)
            term_list_file = backdir + "/" + type + "/term_list"
            all_term_list = read_protein_list(term_list_file)


            name_list = []
            for protein_id in protein_list:
                if (protein_id in label_dict):
                    name_list.append(protein_id)

            name_list_file = workdir + "/" + type.lower() + "_" + data_type + "_used_pid_list.pkl"
            with open(name_list_file, 'wb') as fw:
                pkl.dump(name_list, fw)

            go_label_file = workdir + "/" + type.lower() + "_" + data_type + "_go.txt"

            f = open(go_label_file, "w")

            for name in name_list:
                term_list = label_dict[name]
                term_list = list(set(term_list)&set(all_term_list))
                if(len(term_list)==0):
                    print(name)
                for term in term_list:
                    f.write(name + "\t" + term + "\t" + type.lower() + "\n")
            f.close()

            f = open(workdir + "/" + type.lower() + "_" + data_type + ".fasta", "w")
            for name in name_list:
                f.write(">" + name + "\n" + sequence_dict[name] + "\n")
            f.close()

            temp_pdb_dir = workdir + "/PDB_folder/" + type.lower() + "/" + data_type + "/"
            os.system("rm -rf " + temp_pdb_dir)
            os.makedirs(temp_pdb_dir)
            
            for name in name_list:
                os.system("cp " + pdb_dir + "/" + name + ".pdb " + temp_pdb_dir + "/" + name + ".pdb ")


create_label(sys.argv[1])



