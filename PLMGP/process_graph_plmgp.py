import joblib
import numpy as np
import scipy.sparse as ssp
from pathlib import Path
from collections import defaultdict
from Bio import SeqIO
from tqdm.auto import tqdm, trange
import math
import pickle as pkl
import dgl
import torch
from ruamel.yaml import YAML

import click

prottrans_feature_dir = "/data/yihengzhu/MKFGO_2.0/resource/benchmark/prottrans_features/"

three_to_one = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U",  # 硒代半胱氨酸
    "PYL": "O",  # 吡咯赖氨酸
}


def read_pkl(file_path):
    with open(file_path,'rb') as fr:
        return pkl.load(fr)

def save_pkl(file_path, val):
    fw = open(file_path, 'wb')
    pkl.dump(val, fw)
    fw.close()

def read_protrans_feature(feature_file):   # read features from transformer

    feature = torch.load(feature_file)

    return feature


def get_dis(point1, point2):
    dis_x = point1[0] - point2[0]
    dis_y = point1[1] - point2[1]
    dis_z = point1[2] - point2[2]
    return math.sqrt(dis_x*dis_x + dis_y*dis_y + dis_z*dis_z)

def get_amino_feature(amino):
    # all_for_assign = np.loadtxt("all_assign.txt")
    if amino == 'ALA':
        return 0
    elif amino == 'CYS':
        return 1
    elif amino == 'ASP':
        return 2
    elif amino == 'GLU':
        return 3
    elif amino == 'PHE':
        return 4
    elif amino == 'GLY':
        return 5
    elif amino == 'HIS':
        return 6
    elif amino == 'ILE':
        return 7
    elif amino == 'LYS':
        return 8
    elif amino == 'LEU':
        return 9
    elif amino == 'MET':
        return 10
    elif amino == 'ASN':
        return 11
    elif amino == 'PRO':
        return 12
    elif amino == 'GLN':
        return 13
    elif amino == 'ARG':
        return 14
    elif amino == 'SER':
        return 15
    elif amino == 'THR':
        return 16
    elif amino == 'VAL':
        return 17
    elif amino == 'TRP':
        return 18
    elif amino == 'TYR':
        return 19
    else:
        print("Amino False!")

def read_sequence(sequence_file):  # read sequence

    sequence_dict = dict()

    f = open(sequence_file, "r")
    text = f.read()
    f.close()

    for line in text.splitlines():
        if (line.startswith(">")):
            name = line[1:]
        else:
            line = line.strip()
            sequence_dict[name] = line

    return sequence_dict

def get_whole_pdb_graph(pdb_points, pid_list, sequence_dict, thresholds, ont, tag):

    file_idx = 0
    pdb_graphs = []
    p_cnt = 0

    for pid in tqdm(pid_list):

        p_cnt += 1

        points = pdb_points[pid]

        sequence1 = sequence_dict[pid]
        sequence2 = ""

        u_list = []
        v_list = []
        dis_list = []
        node_amino = {}

        for uid, amino_1 in enumerate(points):
            sequence2 = sequence2 + three_to_one[amino_1[3]]
            for vid, amino_2 in enumerate(points):
                '''
                if uid==vid:
                    continue
                '''
                dist = get_dis(amino_1, amino_2)
                if dist<=thresholds:
                    u_list.append(uid)
                    v_list.append(vid)
                    dis_list.append(dist)

        assert sequence1==sequence2, "sequence error!"
        protrans_feature = read_protrans_feature(prottrans_feature_dir + "/" + pid + ".pt")


        u_list, v_list = torch.tensor(u_list), torch.tensor(v_list)
        dis_list = torch.tensor(dis_list)

        graph = dgl.graph((u_list, v_list), num_nodes=len(points))
        graph.edata['dis'] = dis_list

        # graph node feature
        graph.ndata['x'] = torch.zeros(graph.num_nodes(), 1024)
        graph.ndata['aa'] = torch.zeros(graph.num_nodes(), 20)
        for node_id in range(len(points)):
            amino = points[node_id][3]
            amino_id = get_amino_feature(amino)

            graph.ndata['x'][node_id] = torch.from_numpy(protrans_feature[node_id])

            one_hot = [0.0]*20
            one_hot[amino_id] = 1.0
            graph.ndata['aa'][node_id] = torch.tensor(one_hot)

        pdb_graphs.append(graph)

        if p_cnt%5000==0:
            save_pkl('./data/PDB/graph_feature/{}_{}_whole_pdb_part{}.pkl'.format(ont, tag, file_idx), pdb_graphs)
            p_cnt = 0
            file_idx += 1
            pdb_graphs = []

    if len(pdb_graphs)>0:
        save_pkl('./data/PDB/graph_feature/{}_{}_whole_pdb_part{}.pkl'.format(ont, tag, file_idx), pdb_graphs)

@click.command()
@click.option('-d', '--data-cnf', type=click.Choice(['bp', 'mf', 'cc']), default='mf')
@click.option('-t', '--thresholds', type=click.INT, default=12)

def main(data_cnf, thresholds):

    yaml = YAML(typ='safe')
    data_cnf = yaml.load(Path('./configure/{}.yaml'.format(data_cnf)))
    ont = data_cnf['name']

    data_type_list = ["train", "evaluate", "test"]

    for data_type in data_type_list:

        pid_list_file = data_cnf[data_type]['pid_list_file']
        pdb_points_file = data_cnf[data_type]['pdb_points']
        sequence_file = data_cnf[data_type]['sequence_file']

        with open(pid_list_file, 'rb') as fr:
            used_pid_list = pkl.load(fr)

        with open(pdb_points_file, 'rb') as fr:
            pdb_points = pkl.load(fr)

        sequence_dict = read_sequence(sequence_file)

        print("Used Pid in Test: {}".format(len(used_pid_list)))
        print("Used Pid in Test: {}".format(len(pdb_points)))

        get_whole_pdb_graph(pdb_points, used_pid_list, sequence_dict, thresholds, ont, data_type)



if __name__ == '__main__':
    main()