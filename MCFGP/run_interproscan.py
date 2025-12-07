import os
import sys
import threading
from configure import run_interproscan_file, max_length

def read_sequence(sequence_file):  # read sequence

    sequence_dict = dict()

    f = open(sequence_file, "r")
    text = f.read()
    f.close()

    for line in text.splitlines():
        if (line.startswith(">")):
            name = line[1:]
        else:

            if(len(line)>max_length):
                line = line[0: max_length]
            sequence_dict[name] = line

    return sequence_dict


def split(total_number, thread_number, name_list):   # split_name_list

    name_list=sorted(name_list)
    split_list=[]
    n=total_number/thread_number+1
    for i in range(thread_number+1):
        start=int(i*n)
        end=int((i+1)*n)
        if(end>total_number):
            end=total_number

        if(start<total_number):
            temp_list=[]
            for i in range(start, end, 1):
                temp_list.append(name_list[i])
            split_list.append(temp_list)

    return split_list

def create_script(workdir, thread_number):  # create scripts

    sequence_file = os.path.join(workdir, "all_sequence.fasta")
    sequence_dict = read_sequence(sequence_file)
    name_list = sorted(sequence_dict.keys())
    total_number = len(name_list)
    split_list = split(total_number, thread_number, name_list)

    script_dir = os.path.join(workdir, "script")
    os.system("rm -rf " + script_dir)
    os.makedirs(script_dir)

    sequence_dir = os.path.join(workdir, "sequence")
    os.system("rm -rf " + sequence_dir)
    os.makedirs(sequence_dir)

    result_dir = os.path.join(workdir, "result")
    os.system("rm -rf " + result_dir)
    os.makedirs(result_dir)

    for i in range(len(split_list)):

        index = i + 1
        sub_sequence_file = os.path.join(sequence_dir, "sequence" + str(index) + ".fasta")
        f = open(sub_sequence_file, "w")
        for name in split_list[i]:
            f.write(">" + name + "\n" + sequence_dict[name] + "\n")
        f.close()

        temp_dir = os.path.join(workdir, "temp")
        temp_dir = os.path.join(temp_dir, str(index))
        os.system("rm -rf " + temp_dir)
        os.makedirs(temp_dir)

        result_file = os.path.join(result_dir, "result" + str(index))

        cmd = run_interproscan_file + " -i " + sub_sequence_file + " -f tsv -o " + result_file + " -T " + temp_dir
        script_file = os.path.join(script_dir, "script" + str(index))
        f = open(script_file, "w")
        f.write(cmd + "\n")
        f.close()

def list_split(items, n):
    return [items[i:i+n] for i in range(0, len(items), n)]

def run_single_script(script_dir, current_index_list):

    for index in current_index_list:
        script_file = script_dir + "/script" + str(index)
        f = open(script_file, "r")
        text = f.read()
        f.close()
        os.system(text)

def run_scripts(script_dir, total_number, thread_number):

    index_list = [i for i in range(1, total_number + 1)]
    split_index_list = list_split(index_list, thread_number)

    threads = []

    for current_index_list in split_index_list:
        thread = threading.Thread(target=run_single_script, args=(script_dir, current_index_list,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


if __name__ == '__main__':

    create_script(sys.argv[1], int(sys.argv[2]))
    run_scripts(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))


