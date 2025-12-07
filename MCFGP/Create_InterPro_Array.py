import os.path
import sys
import pickle as pkl
interpro_number = 27517
import numpy as np
from tqdm.auto import trange

interpro_dir = "/data/yihengzhu/toolbars/sequence_homology_tools/InterPro/temps_new/entry_name_new/"
all_entry_file = "/data/yihengzhu/toolbars/sequence_homology_tools/InterPro/temps_new/all_entry_list"


def create_interpro_array(workdir, type):

    data_type_list = ["train", "evaluate", "test"]

    f = open(all_entry_file, "r")
    text = f.read()
    f.close()

    all_entry_list = text.splitlines()

    for data_type in data_type_list:

        protein_list_file = workdir + "/" + type + "_" + data_type + "_used_pid_list.pkl"
        with open(protein_list_file, 'rb') as fr:
            pid_list = pkl.load(fr)

        interpro_array = np.zeros([len(pid_list), interpro_number])

        for i in trange(len(pid_list)):

            pid = pid_list[i]

            interpro_file = interpro_dir + "/" + pid
            if(os.path.exists(interpro_file)==False or os.path.getsize(interpro_file)==0):
                continue

            f = open(interpro_file, "r")
            text = f.read()
            f.close()

            for line in text.splitlines():

                values = line.strip().split()
                index = all_entry_list.index(values[0])
                interpro_array[i][index] = int(values[1])

        save_file = workdir + "/" + type + "_" + data_type + "_interpro.pkl"
        with open(save_file, 'wb') as fw:
            pkl.dump(interpro_array, fw)


create_interpro_array(sys.argv[1], sys.argv[2])