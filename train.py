import json
import time
import os
import random
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import optimization
from sklearn.metrics import accuracy_score
import nni
from nni.utils import merge_parameter
import numpy as np
import pandas as pd
import geopandas as gpd
import transbigdata as tbd
import os.path as osp
from tqdm import tqdm
import pickle
from copy import deepcopy
import fpstmatch.data_preprocess.data_loader_merge as fpdl
import fpstmatch as fp
from fpstmatch.gridroadnet import GridRoadNet
from fpstmatch.model.crf_tf_all import CRF
from fpstmatch.config import get_params
from shapely.geometry import GeometryCollection
from pandarallel import pandarallel
import multiprocessing as mp
from functools import partial
pandarallel.initialize()
warnings.filterwarnings("ignore")


def flatten(li):
    return sum(([x] if not isinstance(x, list) else flatten(x) for x in li), [])


def _edge_grid_union(edge_grid_list):
    if len(edge_grid_list) > 1:
        l = set(edge_grid_list[0])
        for t in edge_grid_list[1:]:
            l = l.union(t)
    else:
        l = set(edge_grid_list[0])

    return fp.gridroadnet.edge_grid_to_gridid(flatten(list(l)), "hexa")


def to_grid_info_exp(sub_route, ignore_true):
    insert_grid = ignore_true[ignore_true['from_grid_str'].str.contains(
        "|".join(sub_route['to_grid']))]
    if len(insert_grid) == 0:
        # geometry_exp = sub_route['geometry']
        # way_prjPt_geom_exp = sub_route['way_prjPt_geom']
        # length_exp = sub_route['length']
        edge_grid_exp = sub_route['edge_grid_extend']
        edge_grid_union = _edge_grid_union(edge_grid_exp)
    else:
        # geometry_exp = GeometryCollection(
        #    [sub_route['geometry']]+list(insert_grid['geometry']))
        # way_prjPt_geom_exp = GeometryCollection(
        #    [sub_route['way_prjPt_geom']]+list(insert_grid['way_prjPt_geom']))
        # length_exp = sub_route['length']+insert_grid['length'].sum()
        edge_grid_exp = sub_route['edge_grid_extend'] + \
            insert_grid['edge_grid_extend'].sum()
        edge_grid_union = _edge_grid_union(edge_grid_exp)

    # return geometry_exp, way_prjPt_geom_exp, length_exp, edge_grid_exp, edge_grid_union
    return edge_grid_exp, edge_grid_union


#def accuracy_threads(idx, _infer_path_list, _object, _pathset, _tlSeg, gRN, sampledtraj_file_path):
def accuracy_threads(args):

    infer_path, _object_, _pathset_, _tlSeg_, gRN, sampledtraj_file_path = args

    def _get_acc(_strseg):
        if infer_path[_strseg.name] == 0:
            return 1.,1.
        else:
            infer_path[_strseg.name] = infer_path[_strseg.name]-1
        try:
            sp = _strseg['odpaths'][infer_path[_strseg.name]]
        except:
            print("_strseg.name:",_strseg.name)
            print("infer_path[_strseg.name]:",infer_path[_strseg.name])
            print("len(_strseg['odpaths']):",len(_strseg['odpaths']))
            raise
        _route_list = list()
        for u, v in zip(sp[:-1], sp[1:]):
            try:
                _route_list.append(list(gRN.G.get_edge_data(u, v).values())[0])
            except:
                fp.logs.info(f"u:{u},v:{v}")
        _route_gdf = gpd.GeoDataFrame(pd.DataFrame(_route_list))
        # _route_gdf[['geometry_exp', 'way_prjPt_geom_exp', 'length_exp', 'edge_grid_exp', 'edge_grid_union']] = _route_gdf.apply(
        #    lambda sub_route: to_grid_info_exp(sub_route, gRN.network_grid_cl_edges_info_ignore_true), axis=1, result_type='expand')
        _route_gdf[['edge_grid_exp', 'edge_grid_union']] = _route_gdf.apply(
            lambda sub_route: to_grid_info_exp(sub_route, gRN.network_grid_cl_edges_info_ignore_true), axis=1, result_type='expand')

        _route_grids = list(_route_gdf['edge_grid_exp'])
        _true_traj_grids = [str(loncol[0])+'_'+str(loncol[1])+'_'+str(loncol[2])
                            for loncol in list(_strseg['true_traj_gridslist'])]
        _precision = len(set(flatten(_route_grids)).intersection(
            set(_true_traj_grids)))/len(set(flatten(_route_grids)))
        
        _recall = len(set(flatten(_route_grids)).intersection(
            set(_true_traj_grids)))/len(set(_true_traj_grids))

        return _precision, _recall

    strseg_path = osp.join(sampledtraj_file_path, _object_, _pathset_,
                           f"{_object_}_{_pathset_}_{_tlSeg_}_strseg.pkl")  # type: ignore

    strseg = pickle.load(open(strseg_path, "rb"))
    strseg.drop(strseg[strseg['odpairs'] == 'ignore'].index, inplace=True)
    strseg.reset_index(drop=True, inplace=True)

    precision, recall = strseg.apply(lambda _strseg: _get_acc(_strseg), axis=1, result_type='expand').mean(axis=0)
    
    return [precision, recall]


def train(model, train_iter, optimizer, scheduler, device, writer=None):
    model.train()
    train_l_sum, count = 0., 0
    for data in tqdm(train_iter):
        dist_nsp = data[0].to(device)
        dur_nsp_seconds = data[1].to(device)
        path_speed = data[2].to(device)
        path_delta_len_norm = data[3].to(device)
        path_dir_norm = data[4].to(device)
        path_det_norm = data[5].to(device)
        path_timeslot_norm = data[6].to(device)
        path_gridrange_norm = data[7].to(device)
        traces_lens = data[8]
        ts_ks = data[9]
        _object = data[10]
        _pathset = data[11]
        _tlSeg = data[12]

        batch_mask = torch.zeros_like(path_speed).to(device)
        batch_mask[torch.where(path_speed != -1)] = True
        batch_mask = batch_mask.bool()
        seq_mask = batch_mask[:, :, 0]

        dist_nsp = torch.masked_fill(dist_nsp, ~seq_mask, 0)
        dur_nsp_seconds = torch.masked_fill(dur_nsp_seconds, ~seq_mask, 0)
        path_speed = torch.masked_fill(path_speed, ~batch_mask, float('nan'))
        path_delta_len_norm = torch.masked_fill(path_delta_len_norm, ~batch_mask, 0)
        path_dir_norm = torch.masked_fill(path_dir_norm, ~batch_mask.unsqueeze(-1), 0)
        path_det_norm = torch.masked_fill(path_det_norm, ~batch_mask.unsqueeze(-1), 0)
        path_timeslot_norm = torch.masked_fill(path_timeslot_norm, ~seq_mask.unsqueeze(-1), 0)
        path_gridrange_norm = torch.masked_fill(path_gridrange_norm, ~seq_mask.unsqueeze(-1), 0)

        loss = model(path_delta_len_norm,
                       path_dir_norm,
                       path_det_norm,
                       path_timeslot_norm,
                       path_gridrange_norm,
                       dist_nsp,
                       dur_nsp_seconds,
                       path_speed,
                       batch_mask)
        if writer is not None:
            writer.add_scalar('Training/Loss', loss, scheduler.last_epoch)
            writer.add_scalar('Training/Learning Rate', scheduler.get_lr()[0], scheduler.last_epoch)
        train_l_sum += loss.item()
        count += 1
        #if count % 10 == 0:
        #    print(f"Iteration {count}: train_loss {loss.item()}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()#retain_graph=True
        #nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        scheduler.step()

    return train_l_sum / count


def evaluate(model, eval_iter, device, sampledtraj_file_path, gRN, e=None, writer=None):
    model.eval()
    count, eval_precision_sum, eval_recall_sum = 0., 0., 0.
    infoshow_pd = pd.DataFrame()
    step=e
    with torch.no_grad():
        for data in tqdm(eval_iter):
            dist_nsp = data[0].to(device)
            dur_nsp_seconds = data[1].to(device)
            path_speed = data[2].to(device)
            path_delta_len_norm = data[3].to(device)
            path_dir_norm = data[4].to(device)
            path_det_norm = data[5].to(device)
            path_timeslot_norm = data[6].to(device)
            path_gridrange_norm = data[7].to(device)
            traces_lens = data[8]
            ts_ks = data[9]
            _object = data[10]
            _pathset = data[11]
            _tlSeg = data[12]

            batch_mask = torch.zeros_like(path_speed).to(device)
            batch_mask[torch.where(path_speed != -1)] = True
            batch_mask = batch_mask.bool()
            seq_mask = batch_mask[:,:,0]

            dist_nsp = torch.masked_fill(dist_nsp, ~seq_mask, 0)
            dur_nsp_seconds = torch.masked_fill(dur_nsp_seconds, ~seq_mask, 0)
            path_speed = torch.masked_fill(path_speed, ~batch_mask, float('nan'))
            path_delta_len_norm = torch.masked_fill(path_delta_len_norm, ~batch_mask, 0)
            path_dir_norm = torch.masked_fill(path_dir_norm, ~batch_mask.unsqueeze(-1), 0)
            path_det_norm = torch.masked_fill(path_det_norm, ~batch_mask.unsqueeze(-1), 0)
            path_timeslot_norm = torch.masked_fill(path_timeslot_norm, ~seq_mask.unsqueeze(-1), 0)
            path_gridrange_norm = torch.masked_fill(path_gridrange_norm, ~seq_mask.unsqueeze(-1), 0)


            infer_seq = model.infer(path_delta_len_norm,
                                    path_dir_norm,
                                    path_det_norm,
                                    path_timeslot_norm,
                                    path_gridrange_norm,
                                    dist_nsp,
                                    dur_nsp_seconds,
                                    path_speed,
                                    batch_mask)
            
            count += 1
            results = []
            for idx in range(len(_object)):
                    results.append(accuracy_threads((infer_seq[idx], _object[idx], _pathset[idx], _tlSeg[idx],gRN,sampledtraj_file_path)))
            p_r = torch.tensor(results).mean(axis=0)
            eval_precision_sum += p_r[0].item()
            eval_recall_sum += p_r[1].item()

            infoshow_pd = pd.concat([infoshow_pd, pd.DataFrame({"object_pathset_tlSeg":[f"{_object[i]}_{_pathset[i]}_{_tlSeg[i]}" for i in range(len(infer_seq))],
                                "infer_seq":infer_seq})], axis=0)
            
        if writer is not None:
            infoshow = infoshow_pd.sort_values(by="object_pathset_tlSeg").reset_index(drop=True).to_markdown()
            writer.add_text(f"infer_seq_info", infoshow, step)

    return eval_precision_sum / count, eval_recall_sum / count



def main(args, timestamp, jsonfilename):

    fp.logs.info("loading gridroadnet_bj...")
    gRN = GridRoadNet(loadexist=True, exisfile_root='/gridroadnet_bj')
    fp.logs.info("finish loading gridroadnet_bj...")

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    # 设置随机数种子
    setup_seed(20)

    batch_size = args['batch_size']
    dp_file_path = osp.join(args['prcdata_path'], args['prcdata_dp_folder'])
    sampledtraj_file_path = osp.join(args['prcdata_path'], args['prcdata_sampledtraj_folder'])

    fp.logs.info("loading trainset and valset...")
    trainset = fpdl.MyDataset(file_path=dp_file_path, name="train", _object_=args['idpdt_train'])
    train_iter = DataLoader(trainset, num_workers=48, batch_size=batch_size, shuffle=False, collate_fn=fpdl.padding,pin_memory=False)
    valset = fpdl.MyDataset(file_path=dp_file_path, name="val", _object_=args['idpdt_train'])
    val_iter = DataLoader(valset, num_workers=48, batch_size=batch_size, shuffle=False, collate_fn=fpdl.padding,pin_memory=False)
    testset = fpdl.MyDataset(file_path=dp_file_path, name="test", _object_=args['idpdt_train'])
    test_iter = DataLoader(testset, num_workers=48, batch_size=batch_size, shuffle=False, collate_fn=fpdl.padding,pin_memory=False)
    fp.logs.info("finish loading trainset and valset")

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    segE_arg = {
        "delta_L": torch.tensor(2000).to(device),
        "delta_T": torch.tensor(1800).to(device),
    }
    pt_arg = {
        "a": torch.tensor(2).to(device),
        "eta": torch.tensor([0.6, 0.4]).to(device),
        "alpha": torch.tensor(3).to(device),
    }

    crf = CRF(segE_arg=segE_arg,
              pt_arg=pt_arg,
              emb_transformer_dim=args['emb_transformer_dim'],
              device=device)

    crf = crf.to(device)



    if args['exited_model_path'] != "":
        try:
            crf.load_state_dict(torch.load(args['exited_model_path']))
            fp.logs.info("loading model finished!")
        except:
            fp.logs.info("loading model failed!")
            raise
    else:
        def weight_init(m):
            if isinstance(m, nn.Linear):
                nn.init.ones_(m.weight)
            if isinstance(m, nn.TransformerEncoderLayer):
                nn.init.ones_(m.self_attn.in_proj_weight)
        crf.apply(weight_init)

    fp.logs.info("model init finished!")
    best_precision, best_recall, best_model = 0., 0., None
    optimizer = optim.Adam(params=crf.parameters(), lr=args['lr'])
    
    num_training_steps = len(train_iter) * args['epochs']
    #scheduler = optimization.get_cosine_schedule_with_warmup(optimizer, 
    #                                                        num_training_steps=num_training_steps,
    #                                                        num_cycles=2,
    #                                                        num_warmup_steps=30,
    #                                                        last_epoch=-1).get_lr
    scheduler = optimization.get_constant_schedule(optimizer)
    if args['idpdt_train'] is None:
        writer = SummaryWriter(log_dir=f"runs/crf_train_tf_all_{timestamp}_{args['batch_size']}_{args['lr']}_{args['emb_transformer_dim']}", flush_secs=120)
    else:
        writer = SummaryWriter(log_dir=f"runs/crf_train_tf_all_{timestamp}_{args['batch_size']}_{args['lr']}_{args['emb_transformer_dim']}_{args['idpdt_train']}", flush_secs=120)

    fp.logs.info("start training...")
    try:
        for e in range(args['epochs']):
            fp.logs.info(f"================Epoch: {e + 1}================")
            train_avg_loss = train(crf, train_iter, optimizer, scheduler, device, writer)
            writer.add_scalar('Train/Loss', train_avg_loss, scheduler.last_epoch)
            writer.add_scalar('Train/Learning Rate', scheduler.get_last_lr()[0], scheduler.last_epoch)
            precision, recall = evaluate(crf, val_iter, device,sampledtraj_file_path, gRN, e, writer)
            writer.add_scalar('Validation/precision', precision, scheduler.last_epoch)
            writer.add_scalar('Validation/recall', recall, scheduler.last_epoch)
            # choose model based on val_acc
            if best_precision < precision:
                best_model = deepcopy(crf.state_dict())
                best_precision = precision
                best_recall = recall
            fp.logs.info(f"Epoch {e + 1}: train_avg_loss {train_avg_loss} precision_avg: {precision} recall_avg: {recall}")

        fp.logs.info(f"finish training! best_precision: {best_precision} recall_avg: {best_recall}")

        if not os.path.exists("model"):
            os.makedirs("model")
        if args['idpdt_train'] is None:
            torch.save(crf.state_dict(
            ), f"model/{timestamp}_trainedpara_lastepoch_model_tf_all_{args['batch_size']}_{args['lr']}_{args['emb_transformer_dim']}.pt")
            if best_model is not None:
                torch.save(best_model, f"model/{timestamp}_trainedpara_best_model_tf_all_{args['batch_size']}_{args['lr']}_{args['emb_transformer_dim']}.pt")
        else:
            if not os.path.exists(f"model/{args['idpdt_train']}_tf_all"):
                os.makedirs(f"model/{args['idpdt_train']}_tf_all")
            torch.save(crf.state_dict(
            ), f"model/{args['idpdt_train']}_tf_all/{timestamp}_trainedpara_lastepoch_model_tf_all_{args['batch_size']}_{args['lr']}_{args['emb_transformer_dim']}_{args['idpdt_train']}.pt")
            if best_model is not None:
                torch.save(best_model, f"model/{args['idpdt_train']}_tf_all/{timestamp}_trainedpara_best_model_tf_all_{args['batch_size']}_{args['lr']}_{args['emb_transformer_dim']}_{args['idpdt_train']}.pt")
        fp.logs.info(f"lastepoch and best model saved.")

        if best_model is not None:
            fp.logs.info("start testing...")
            crf.load_state_dict(best_model)
            test_precision, test_recall = evaluate(crf, test_iter, device,sampledtraj_file_path, gRN, e, writer)
            writer.add_scalar('Test/precision', test_precision, scheduler.last_epoch)
            writer.add_scalar('Test/recall', test_recall, scheduler.last_epoch)
            fp.logs.info(f"Best model on Test DataSet. test_precision: {test_precision} test_recall: {test_recall}")
            with open(jsonfilename,'r') as f:
                jsonfile = json.load(f)
            jsonfile['actual epochs'] = e+1
            jsonfile['best_precision'] = best_precision
            jsonfile['best_recall'] = best_recall
            jsonfile['test_precision'] = test_precision
            jsonfile['test_recall'] = test_recall
            with open(jsonfilename,'w') as f:
                json.dump(jsonfile,f)
            f.close()
            fp.logs.info(f"results saved to {jsonfilename}") 
        else:
            fp.logs.info("best_model is None")

        writer.close()


    except:
        if not os.path.exists("model"):
            os.makedirs("model")
        if args['idpdt_train'] is None:
            torch.save(crf.state_dict(
            ), f"model/{timestamp}_trainedpara_lastepoch_model_tf_all_{args['batch_size']}_{args['lr']}_{args['emb_transformer_dim']}.pt")
            if best_model is not None:
                torch.save(best_model, f"model/{timestamp}_trainedpara_best_model_tf_all_{args['batch_size']}_{args['lr']}_{args['emb_transformer_dim']}.pt")
        else:
            if not os.path.exists(f"model/{args['idpdt_train']}_tf_all"):
                os.makedirs(f"model/{args['idpdt_train']}_tf_all")
            torch.save(crf.state_dict(
            ), f"model/{args['idpdt_train']}_tf_all/{timestamp}_trainedpara_lastepoch_model_tf_all_{args['batch_size']}_{args['lr']}_{args['emb_transformer_dim']}_{args['idpdt_train']}.pt")
            if best_model is not None:
                torch.save(best_model, f"model/{args['idpdt_train']}_tf_all/{timestamp}_trainedpara_best_model_tf_all_{args['batch_size']}_{args['lr']}_{args['emb_transformer_dim']}_{args['idpdt_train']}.pt")
        fp.logs.info(f"lastepoch and best model saved.")

        if best_model is not None:
            fp.logs.info("start testing...")
            crf.load_state_dict(best_model)
            test_precision, test_recall = evaluate(crf, test_iter, device,sampledtraj_file_path, gRN, e, writer)
            writer.add_scalar('Test/precision', test_precision, scheduler.last_epoch)
            writer.add_scalar('Test/recall', test_recall, scheduler.last_epoch)
            fp.logs.info(f"Best model on Test DataSet. test_precision: {test_precision} test_recall: {test_recall}")
            with open(jsonfilename,'r') as f:
                jsonfile = json.load(f)
            jsonfile['actual epochs'] = e+1
            jsonfile['best_precision'] = best_precision
            jsonfile['best_recall'] = best_recall
            jsonfile['test_precision'] = test_precision
            jsonfile['test_recall'] = test_recall
            with open(jsonfilename,'w') as f:
                json.dump(jsonfile,f)
            f.close()
            fp.logs.info(f"results saved to {jsonfilename}")      
        else:
            fp.logs.info("best_model is None")
        writer.close()

        raise


if __name__ == "__main__":
    timestamp = time.strftime('%Y%m%d%H%M', time.localtime())
    tuner_params = nni.get_next_parameter()
    params = vars(merge_parameter(get_params(), tuner_params))

    params.pop('emb_linear_dim')
    params['model_name'] = 'crf_tf_all'

    # 把params保存到文件中
    if not os.path.exists("model"):
        os.makedirs("model")

    if params['idpdt_train'] == '':
        params['idpdt_train'] = None

    if params['idpdt_train'] is None:
        jsonfilename = f"model/{timestamp}_model_params_{params['batch_size']}_{params['lr']}_{params['emb_transformer_dim']}.json"
        with open(jsonfilename, 'w') as f:
            json.dump(params, f)
        f.close()
    else:
        if not os.path.exists(f"model/{params['idpdt_train']}_tf_all"):
            os.makedirs(f"model/{params['idpdt_train']}_tf_all")
        jsonfilename = f"model/{params['idpdt_train']}_tf_all/{timestamp}_model_params_{params['batch_size']}_{params['lr']}_{params['emb_transformer_dim']}_{params['idpdt_train']}.json"
        with open(jsonfilename, 'w') as f:
            json.dump(params, f)
        f.close()
    
    print(params)
    main(params, timestamp, jsonfilename)
