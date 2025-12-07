import os
import sys
import numpy as np
from Bio.PDB import PDBParser, DSSP

ss_type = ["H" , "G" , "I" , "E" , "B" , "T", "S", "C"]

def run_dssp(pdb_file, result_file, mkdssp_path="/data/yihengzhu/toolbars/strcuture_alignment_tools/dssp-master/mkdssp"):
    if not os.path.isfile(pdb_file):
        raise FileNotFoundError(f"输入文件 {pdb_file} 不存在！")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    model = structure[0]

    # 在这里指定 mkdssp 路径
    dssp = DSSP(model, pdb_file, dssp=mkdssp_path)

    sd_array = []

    # 正确的遍历方式：遍历keys，再用 dssp[key] 取值
    for key in dssp.keys():
        # key 形如: (chain_id, (' ', resseq, icode))
        chain_id = key[0]
        resseq = key[1][1]  # 残基序号
        icode = key[1][2]  # 插入码（可能是' '）
        res_id_str = f"{resseq}{icode.strip()}" if icode.strip() else f"{resseq}"

        vals = dssp[key]
        aa = vals[1]  # 氨基酸单字母
        ss = dssp_to_q8(vals[2])
        asa = vals[3]  # 可及表面积
        phi1, phi2 = angle_to_sincos(float(vals[4]))
        psi1, psi2 = angle_to_sincos(float(vals[5]))

        temp_array = [0 for i in range(8)]
        ss_index = ss_type.index(ss)
        temp_array[ss_index] = 1
        temp_array.extend([asa, phi1, phi2, psi1, psi2])

        sd_array.append(temp_array)

    sd_array = np.array(sd_array)
    np.save(result_file, sd_array)




def dssp_to_q8(ss_raw: str) -> str:
    """
    将 DSSP 原始符号映射到统一的 Q8 集合: {H, B, E, G, I, T, S, C}

    - 合法的 H, B, E, G, I, T, S 保持不变
    - DSSP 输出中的 ' ' (空格)、'-' (未定义/缺残基) -> 'C'
    - 其他未知字符 -> 'C'
    """
    if not ss_raw:
        return "C"
    ss = str(ss_raw).strip().upper()
    if ss in {"H", "B", "E", "G", "I", "T", "S"}:
        return ss
    return "C"

def angle_to_sincos(angle_deg):
    rad = np.deg2rad(angle_deg)
    return np.cos(rad), np.sin(rad)

if __name__ == "__main__":

    structure_dir = sys.argv[1]
    psd_dir = sys.argv[2]

    name_list = os.listdir(structure_dir)

    for name in name_list:
        run_dssp(structure_dir + "/" + name, psd_dir + "/" + name.split(".")[0] + ".npy")



