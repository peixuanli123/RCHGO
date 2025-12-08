import sys

from ruamel.yaml import YAML
from logzero import logger
from pathlib import Path
import warnings

import torch
import numpy as np
from dgl.dataloading import GraphDataLoader

from data_utils import get_pdb_data, get_mlb, read_interpro
from models_ppsi_gog import combine_inter_model
from objective import AverageMeter
from model_utils_ppsi_gog import test_performance_gnn_inter, merge_result, FocalLoss

import os
from tqdm.auto import tqdm
import Create_Results as cr


def main(data_cnf, gpu_number, epoch_number, round_index):
    yaml = YAML(typ='safe')
    ont = data_cnf
    data_cnf, model_cnf = yaml.load(Path('./configure/{}.yaml'.format(data_cnf))), yaml.load(Path('./configure/dgg.yaml'))
    device = torch.device('cuda:{}'.format(gpu_number))

    data_name, model_name = data_cnf['name'], model_cnf['name'] 
    run_name = F'{model_name}-{data_name}'
    logger.info('run_name: {}'.format(run_name))

    data_cnf['mlb'] = Path(data_cnf['mlb'])
    data_cnf['results'] = Path(data_cnf['results'])
    logger.info(F'Model: {model_name}, Dataset: {data_name}')

    train_pid_list, train_graph, train_go = get_pdb_data(pid_list_file = data_cnf['train']['pid_list_file'],
                                                         pdb_graph_file = data_cnf['train']['pid_pdb_file'],
                                                         pid_go_file = data_cnf['train']['pid_go_file'], 
                                                         train = data_cnf['train']['train_file_count'])


    logger.info('train data done')
    valid_pid_list, valid_graph, valid_go = get_pdb_data(pid_list_file = data_cnf['evaluate']['pid_list_file'],
                                                         pdb_graph_file = data_cnf['evaluate']['pid_pdb_file'],
                                                         pid_go_file = data_cnf['evaluate']['pid_go_file'])
    logger.info('valid data done')
    test_pid_list, test_graph, test_go = get_pdb_data(pid_list_file = data_cnf['test']['pid_list_file'],
                                                      pdb_graph_file = data_cnf['test']['pid_pdb_file'],
                                                      pid_go_file = data_cnf['test']['pid_go_file'])
    logger.info('test data done')

    train_interpro, valid_interpro, test_interpro = read_interpro(data_cnf['train']['interpro_file'], data_cnf['evaluate']['interpro_file'], data_cnf['test']['interpro_file'])

    assert len(train_pid_list) == len(train_graph)
    assert len(train_pid_list) == train_interpro.shape[0]
    assert len(train_pid_list) == len(train_go)

    assert len(valid_pid_list) == len(valid_graph)
    assert len(valid_pid_list) == valid_interpro.shape[0]
    assert len(valid_pid_list) == len(valid_go)

    assert len(test_pid_list) == len(test_graph)
    assert len(test_pid_list) == test_interpro.shape[0]
    assert len(test_pid_list) == len(test_go)

    gog_matrix = np.load(data_cnf['gog'])
    gog_tensor = torch.from_numpy(gog_matrix)
    gog_tensor = gog_tensor.to(device)

    mlb = get_mlb(Path(data_cnf['mlb']), train_go)
    labels_num = len(mlb.classes_)
    print(labels_num)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        train_y = mlb.transform(train_go).astype(np.float32)
        valid_y = mlb.transform(valid_go).astype(np.float32)
        test_y  = mlb.transform(test_go).astype(np.float32)

    idx_goid = {}
    goid_idx = {}
    for idx, goid in enumerate(mlb.classes_):
        idx_goid[idx] = goid
        goid_idx[goid] = idx
        
    train_data = [(train_graph[i], i, train_y[i]) for i in range(len(train_y))]
    train_dataloader = GraphDataLoader(
        train_data,
        batch_size=32,
        drop_last=False,
        shuffle=True)

    valid_data = [(valid_graph[i], i, valid_y[i]) for i in range(len(valid_y))]
    valid_dataloader = GraphDataLoader(
        valid_data,
        batch_size=64,
        drop_last=False,
        shuffle=False)

    test_data = [(test_graph[i], i, test_y[i]) for i in range(len(test_y))]
    test_dataloader = GraphDataLoader(
        test_data,
        batch_size=64,
        drop_last=False,
        shuffle=False)

    del train_graph
    del test_graph
    del valid_graph
    
    logger.info('Loading Data & Model')

    model = combine_inter_model(graph_size1=20, graph_size2=8, graph_size3=5, interpro_size=27517, embedding_hid=128, graph_hid=1024, label_num=labels_num, head=8).to(device)
    logger.info(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4)
    loss_fn = FocalLoss()

    for e in range(1, epoch_number + 1):
        model.train()
        train_loss_vals = AverageMeter()
        for batched_graph, sample_idx, labels in tqdm(train_dataloader, leave=False):
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)
            feats1 = batched_graph.ndata['x']
            feats2 = batched_graph.ndata['y'][:, 0:8]
            feats3 = batched_graph.ndata['y'][:, 8: ]

            inter_features = torch.from_numpy(train_interpro[sample_idx]).to(device).float()

            logits = model(batched_graph, feats1, feats2, feats3, inter_features, gog_tensor)

            loss = loss_fn(logits, labels)
            train_loss_vals.update(loss.item(), len(labels))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if(e<=10):
            continue

        model_name = "cross_entropy_result"
        result_dir = "./results/" + ont.upper() + "/" + model_name + "/evaluate/"
        if(os.path.exists(result_dir)==False):
            os.makedirs(result_dir)

        result_dir = "./results/" + ont.upper() + "/" + model_name + "/test/"
        if (os.path.exists(result_dir) == False):
            os.makedirs(result_dir)

        evaluate_result_file = './results/{0}/{1}/{2}/{3}_{4}_result.pkl'.format(ont.upper(), model_name, "evaluate", round_index, e)
        test_result_file = './results/{0}/{1}/{2}/{3}_{4}_result.pkl'.format(ont.upper(), model_name, "test", round_index, e)

        evaluate_result_dir = './results/{0}/{1}/{2}/round{3}/result{4}/'.format(ont.upper(), model_name, "evaluate", round_index, e)
        test_result_dir = './results/{0}/{1}/{2}/round{3}/result{4}/'.format(ont.upper(), model_name, "test", round_index, e)

        evaluate_label_file = "./results/{0}/{1}_gene_label".format(ont.upper(), "evaluate")
        test_label_file = "./results/{0}/{1}_gene_label".format(ont.upper(), "test")

        record_file = "./results/{0}/record_{1}".format(ont.upper(), round_index)

        roc_dir = "./results/{0}/{1}_roc/".format(ont.upper(), "cross_entropy")
        if(os.path.exists(roc_dir)==False):
            os.mkdir(roc_dir)

        evaluate_roc_file = roc_dir + "/roc_" + str(round_index) + "_" + str(e) + "_evaluate"
        test_roc_file = roc_dir + "/roc_" + str(round_index) + "_" + str(e) + "_test"
        
        valid_df = test_performance_gnn_inter(model, valid_dataloader, valid_interpro, gog_matrix, valid_pid_list, valid_y, idx_goid, goid_idx, ont, device, save=True, save_file=evaluate_result_file, evaluate=False)
        cr.create_result(evaluate_result_file, evaluate_result_dir, ont.upper(), "cross_entropy", evaluate_label_file, evaluate_roc_file, record_file, e, "evaluate")

        test_df = test_performance_gnn_inter(model, test_dataloader, test_interpro, gog_matrix, test_pid_list, test_y, idx_goid, goid_idx, ont, device, save=True, save_file=test_result_file, evaluate=False)
        cr.create_result(test_result_file, test_result_dir, ont.upper(), "cross_entropy", test_label_file, test_roc_file, record_file, e, "test")

        #logger.info('Epoch: {}, Train Loss: {:.6f}\tValid Loss: {:.6f}, plus_Fmax on valid: {:.4f}, AUPR on valid: {:.4f}, cut-off: {:.2f}, df_shape: {}'.format(e, train_loss_vals.avg, valid_loss_avg, valid_plus_fmax, valid_plus_aupr, valid_plus_t, valid_df.shape))
        #logger.info('Epoch: {}, Train Loss: {:.6f}\tTest Loss: {:.6f}, plus_Fmax on test: {:.4f}, AUPR on valid: {:.4f}, cut-off: {:.2f}, df_shape: {}'.format(e, train_loss_vals.avg, test_loss_avg, test_plus_fmax, test_plus_aupr, test_plus_t, test_df.shape))





        model_dir = "./results/" + ont.upper() + "/model/" + str(round_index) + "/"
        if(os.path.exists(model_dir)==False):
            os.makedirs(model_dir)

        '''
        torch.save({'epoch': e, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   './results/{0}/model/{1}/model_{2}.pt'.format(ont.upper(), round_index, e))
                   '''
        logger.info("\t\t\t\t\tSave")




if __name__ == '__main__':

    epoch_number = 30
    for round_index in range(1, 6):
        main(sys.argv[1], sys.argv[2], epoch_number, round_index)

