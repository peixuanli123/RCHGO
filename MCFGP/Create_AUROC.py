import sys
import os
ic_list_file = "/data/yihengzhu/MKFGO_2.0/GOA/ic_list"

def read_ic_list():  # read information content

    f = open(ic_list_file, "r")
    text = f.read()
    f.close()

    ic_dict = dict()

    for line in text.splitlines():
        line = line.strip()
        value = line.split()
        ic_dict[value[0]] = float(value[1])

    return ic_dict

ic_dict = read_ic_list()

def read_term_frequency(train_label_file):

    f = open(train_label_file, "r")
    text = f.read()
    f.close()

    term_dict = dict()

    for line in text.splitlines():
        term_list = line.strip().split()[1].split(",")
        for term in term_list:
            if(term not in term_dict):
                term_dict[term] = 0
            term_dict[term] = term_dict[term] + 1

    return term_dict

backdir = "/data/yihengzhu/GOA/resource/benchmark/back/"
type_list = ["MF", "BP", "CC"]
all_term_dict = dict()
for type in type_list:
    all_term_dict[type] = read_term_frequency(backdir + "/" + type + "/train_gene_label")


def read(labelfile, type):  # read label

    label_dict = dict()
    f=open(labelfile,"r")
    line_txt = f.read()
    f.close()

    all_term_list = []
    name_list = []

    for line in line_txt.splitlines():
        values = line.strip().split()
        term_list = values[1].split(",")
        label_dict[values[0]] = term_list
        all_term_list.extend(term_list)
        name_list.append(values[0])

    all_term_list = list(set(all_term_list))

    new_term_list = []
    for term in all_term_list:
        if(term in all_term_dict[type] and all_term_dict[type][term]>10):
            new_term_list.append(term)

    return label_dict, name_list, new_term_list

def read_result(resultfile): #read results

    result_dict = dict()
    if (os.path.exists(resultfile) == False):
        return result_dict


    f=open(resultfile,"r")
    line_txt = f.read()
    f.close()

    for line in line_txt.splitlines():

        prob = float(line.split()[2])
        if(prob>0):
            result_dict[line.split()[0]] = prob

    return result_dict

def create_single_auc(score_array, label_array):

    pos_score_list = []
    neg_score_list = []

    for i in range(len(score_array)):
        if (label_array[i] == 1):
            pos_score_list.append(score_array[i])
        else:
            neg_score_list.append(score_array[i])

    sum1 = 0.0
    for i in range(len(pos_score_list)):
        for j in range(len(neg_score_list)):
            if (pos_score_list[i] > neg_score_list[j]):
                sum1 = sum1 + 1
            if (pos_score_list[i] == neg_score_list[j]):
                sum1 = sum1 + 0.5
    auc_value = sum1 / (len(pos_score_list) * len(neg_score_list))

    return auc_value

def get_result(resultdir, resultfile_name):  # read result

    result_list_dict = dict()
    list_dir = os.listdir(resultdir)

    for name in list_dir:
        result_list_dict[name] = read_result(resultdir + "/" + name + "/" + resultfile_name)

    return result_list_dict

def median_score(score_list):  # create median score

    length = len(score_list)
    score_list = sorted(score_list, reverse=True)

    if (length % 2) == 1:

        index = int(length/2)
        score = score_list[index]
    else:

        score = (score_list[int(length/2)] + score_list[int(length/2) - 1])/2

    return score

def create_auc(label_dict, result_dict, name_list, all_term_list):

    number = len(name_list)
    average_auc = 0
    sum = 0

    for term in all_term_list:

        label_array = [0 for i in range(number)]
        score_array = [0 for i in range(number)]


        for i in range(number):

            if(term in label_dict[name_list[i]]):
                label_array[i] = 1

            if(name_list[i] in result_dict and term in result_dict[name_list[i]]):
                score_array[i] = float(result_dict[name_list[i]][term])

        auc =  create_single_auc(score_array, label_array)
        average_auc = average_auc + auc * ic_dict[term]
        sum = sum + ic_dict[term]

    average_auc = average_auc/sum

    return average_auc







if __name__=="__main__":

    workdir = sys.argv[1]

    method_list = ["sagp", "funfams", "ppi", "deepgocnn", "tale", "deepgozero", "annopro", "hand_triplet", "atgo", "deepgose", "dpfunc", "prottrans", "deepgoplus", "tale_plus", "atgo_plus", "mkfgo_new"]
    postfix_list = ["protein_Result", "funfam", "ppi_result", "deepgocnn", "tale", "deepgozero", "annopro", "final_combine", "final_combine", "deepgose", "dpfunc", "final_cross_entropy", "deepgoplus", "tale_plus", "atgo_plus", "final_cross_entropy"]

    data_type = "test"
    type_list = ["MF", "BP", "CC"]

    result_dict = dict()

    for i in range(len(method_list)):

        print(method_list[i])

        line = ""

        for type in type_list:

            result_dir = workdir + "/" + method_list[i] + "/" + type + "/" + data_type + "/"
            label_file = workdir + "/" + method_list[i] + "/" + type + "/" + data_type + "_gene_label"

            label_dict, name_list, all_term_list = read(label_file, type)
            result_list_dict = get_result(result_dir, postfix_list[i] + "_" + type + "_new")
            average_auc = create_auc(label_dict, result_list_dict, name_list, all_term_list)
            line = line + str(round(average_auc,3)) + " "

        line = line.strip()
        print(line)
