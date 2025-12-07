import os.path
import sys
from configure import run_interproscan_file, all_entry_list_file
import numpy as np
def read_result_file(result_file):  # read result.file

    f = open(result_file, "r")
    text = f.read()
    f.close()

    result_dict = dict()

    for line in text.splitlines():
        values = line.split("	")
        name = values[0]
        entry = values[11]

        if(name not in result_dict):
            result_dict[name] = []

        result_dict[name].append(entry)

    for name in result_dict.keys():

        temp_list = sorted(list(set(result_dict[name])))
        if("-" in temp_list):
            temp_list.remove("-")

        result_dict[name] = temp_list

    return result_dict

def read_all_entry_list(entry_list_file):  # read all entry list

    f = open(entry_list_file, "r")
    text = f.read()
    f.close()

    all_entry_list = []
    line_set = text.splitlines()
    line_set = line_set[1:]

    for line in line_set:
        all_entry_list.append(line.split("	")[0])

    return all_entry_list

def create_entry_array(current_entry_list, all_entry_list): # create entry index

    entry_number = len(all_entry_list)
    entry_array = np.array([0 for i in range(entry_number)])
    for entry in current_entry_list:
        index = all_entry_list.index(entry)
        entry_array[index] = 1

    return entry_array

def write_file(array_list, save_file):

    f = open(save_file, "w")
    for array in array_list:
        f.write(str(array) + "\n")
    f.close()

def single_process(result_file, output_dir1, output_dir2):

    if(os.path.exists(output_dir1)==False):
        os.makedirs(output_dir1)

    if (os.path.exists(output_dir2) == False):
        os.makedirs(output_dir2)

    result_dict = read_result_file(result_file)
    all_entry_list = read_all_entry_list(all_entry_list_file)

    for name in result_dict.keys():

        entry_name_file = os.path.join(output_dir1, name)
        entry_array_file = os.path.join(output_dir2, name) + ".npy"

        entry_name_list = result_dict[name]
        entry_array = create_entry_array(entry_name_list, all_entry_list)

        write_file(entry_name_list, entry_name_file)
        np.save(entry_array_file, entry_array)

def create_rest_features(sequence_file, output_dir1, output_dir2):

    f = open(sequence_file, "r")
    text = f.read()
    f.close()

    for line in text.splitlines():
        line = line.strip()
        if(line.startswith(">")):

            name = line[1:]

            entry_name_file = os.path.join(output_dir1, name)
            entry_array_file = os.path.join(output_dir2, name) + ".npy"

            f = open(entry_name_file, "w")
            f.close()

            all_entry_list = read_all_entry_list(all_entry_list_file)
            entry_number = len(all_entry_list)

            entry_array = np.array([0 for i in range(entry_number)])
            np.save(entry_array_file, entry_array)





if __name__ == '__main__':

    result_dir = sys.argv[1]
    for i in range(1, 101):
        result_file = result_dir + "/result" + str(i)
        single_process(result_file, sys.argv[2], sys.argv[3])
    

    #create_rest_features(sys.argv[1], sys.argv[2], sys.argv[3])








