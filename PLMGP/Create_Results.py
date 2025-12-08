
import sys
import os
import pickle as pkl

def read_result(result_file):

    with open(result_file, 'rb') as fr:
        result_data = pkl.load(fr)

    protein_id_list = result_data["protein_id"]
    go_term_list = result_data["gos"]
    pro_list = result_data["predictions"]

    return protein_id_list, go_term_list, pro_list

def create_result(origin_result_file, result_dir, type, model_name, label_file, roc_file, record_file, index, data_type):

    protein_id_list, go_term_list, pro_list = read_result(origin_result_file)

    for i in range(len(protein_id_list)):
        protein_id = protein_id_list[i]
        temp_result_dir = result_dir + "/" + protein_id + "/"
        os.system("rm -rf " + temp_result_dir)
        os.makedirs(temp_result_dir)

        result_file = temp_result_dir + "/" + model_name + "_" + type
        result_dict = pro_list[i]
        f = open(result_file, "w")
        for term in result_dict:
            if (result_dict[term] >= 0.05):
                f.write(term + " " + type[1] + " " + str(round(result_dict[term], 3)) + "\n")
        f.close()

    os.system("python2 /data/yihengzhu/MKFGO_2.0/pythonfile/Find_Parents.py " + result_dir +  " " + type + " " + model_name)
    os.system("python /data/yihengzhu/MKFGO_2.0/pythonfile/Evaluate_Single_Pipeline.py " + type + " " + data_type + " " + model_name + " " + label_file + " " + result_dir + " " + roc_file + " " + record_file + " " + str(index))


