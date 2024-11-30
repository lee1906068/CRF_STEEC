import pickle
import os.path as osp
import torch
import pandas as pd
import multiprocessing as mp
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, file_path, name, _object_=None):
        # file_path = "/home/zhengyuanbo/project/FPSTMatch/data_preprocess/data/"
        # name = "train" or "val" or "test"
        # _object_ = "000" if only one object is needed to be loaded for training else None

        super(MyDataset, self).__init__()

        if not file_path.endswith('/'):
            file_path += '/'
        self.name = name
        self._object_ = _object_
        self.data_path = osp.join(
            file_path, f"{self.name}_data/{self.name}set.pkl")
        self.data_pkl = self.buildingDataset()
    def buildingDataset(self):
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)

        data.reset_index(inplace=True)
        if self._object_ is not None:
            data = data[data['object'] == self._object_]

        groups = data.groupby(["object", "pathset", "tlSeg"])
        # groups = data.groupby(["object", "pathset"])

        pool = mp.Pool(processes=mp.cpu_count())
        self.trace_feature = pool.map(self.getfeature, groups)
        pool.close()

        self.dist_nsp_ls = [self.trace_feature[l][0]
                            for l in range(len(self.trace_feature))]
        self.dur_nsp_seconds_ls = [self.trace_feature[l][1]
                                   for l in range(len(self.trace_feature))]
        self.path_speed_ls = [self.trace_feature[l][2]
                              for l in range(len(self.trace_feature))]
        self.dist_delta_ls = [self.trace_feature[l][3]
                              for l in range(len(self.trace_feature))]
        self.path_dir_ls = [self.trace_feature[l][4]
                            for l in range(len(self.trace_feature))]
        self.path_det_ls = [self.trace_feature[l][5]
                            for l in range(len(self.trace_feature))]
        self.path_timeslot_ls = [self.trace_feature[l][6]
                                 for l in range(len(self.trace_feature))]
        self.path_gridrange_ls = [self.trace_feature[l][7]
                                  for l in range(len(self.trace_feature))]

        self._object = [self.trace_feature[l][8]
                        for l in range(len(self.trace_feature))]
        self._pathset = [self.trace_feature[l][9]
                         for l in range(len(self.trace_feature))]
        self._tlSeg = [self.trace_feature[l][10]
                       for l in range(len(self.trace_feature))]
        self._true_traj_ls = [self.trace_feature[l][11]
                              for l in range(len(self.trace_feature))]
        self.path_true_traj_gridslist_ls = [self.trace_feature[l][12]
                                            for l in range(len(self.trace_feature))]
        self.path_det_area_ls = [self.trace_feature[l][13]
                                 for l in range(len(self.trace_feature))]
        self.path_det_name_ls = [self.trace_feature[l][14]
                                 for l in range(len(self.trace_feature))]

        self.length = len(self.dist_nsp_ls)

        return data
    
    def getfeature(self, group):
        group_name, group_data = group
        dist_nsp_ls = group_data['dist_nsp'].round(2).tolist()
        dur_nsp_seconds_ls = group_data['dur_nsp_seconds'].round(2).tolist()
        path_speed_ls = group_data['path_speed'].tolist()
        dist_delta_ls = group_data['path_delta_len_norm'].tolist()
        path_dir_ls = [list(map(list, zip(group_data['path_dir_head_norm'].values[i],group_data['path_dir_turn_norm'].values[i]))) for i in range(len(group_data))]
        path_det_ls = [list(map(list, zip(group_data['path_det_count_norm'].values[i],group_data['path_det_dist_norm'].values[i],group_data['path_det_len_norm'].values[i]))) for i in range(len(group_data))]
        path_timeslot_ls = group_data[[
            'time_o_slot_norm', 'time_slot_incr_norm']].values.tolist()
        path_gridrange_ls = group_data[['tlseg_loncol_1_min_norm', 'tlseg_loncol_1_incr_norm', 'tlseg_loncol_2_min_norm',
                                        'tlseg_loncol_2_incr_norm', 'tlseg_loncol_3_min_norm', 'tlseg_loncol_3_incr_norm']].values.tolist()
        path_true_traj_ls = group_data['true_traj'].astype(str).values.tolist()
        path_true_traj_gridslist_ls = group_data['gridslist'].values.tolist()
        path_det_name_ls = group_data[['det_name_o', 'det_name_d']].astype(
            str).values.tolist()
        path_det_area_ls = group_data[['det_area_o', 'det_area_d']].astype(
            str).values.tolist()

        return dist_nsp_ls, dur_nsp_seconds_ls, path_speed_ls, dist_delta_ls, path_dir_ls, path_det_ls, path_timeslot_ls, path_gridrange_ls, group_name[0], group_name[1], group_name[2], path_true_traj_ls, path_true_traj_gridslist_ls, path_det_area_ls, path_det_name_ls

    def __getitem__(self, index):
        return self.dist_nsp_ls[index],self.dur_nsp_seconds_ls[index], self.path_speed_ls[index],\
                self.dist_delta_ls[index], self.path_dir_ls[index], self.path_det_ls[index], \
                self.path_timeslot_ls[index], self.path_gridrange_ls[index], \
                self._object[index], self._pathset[index], self._tlSeg[index], self._true_traj_ls[index], \
                self.path_true_traj_gridslist_ls[index], self.path_det_area_ls[index], self.path_det_name_ls[index]


    def __len__(self):
        return self.length


def padding(batch):
    trace_lens = [len(sample[0]) for sample in batch]
    max_tlen = max(trace_lens)
    ts_ks = [len(sample[2][i]) for sample in batch for i in range(len(sample[2]))]
    K_max = max(ts_ks)
    dist_nsp_ls, dur_nsp_seconds_ls, path_speed_ls, dist_delta_ls, path_dir_ls, path_det_ls, path_timeslot_ls, path_gridrange_ls = [], [], [], [], [], [], [], []
    for sample in batch:
        dist_nsp_ls.append(sample[0] + [-1] * (max_tlen - len(sample[0])))
        dur_nsp_seconds_ls.append(sample[1] + [-1] * (max_tlen - len(sample[1])))
        path_speed_ls.append([sample[2][i]+[-1]*(K_max-len(sample[2][0])) for i in range(len(sample[2]))] + [[-1]*K_max] * (max_tlen - len(sample[2])))
        dist_delta_ls.append([sample[3][i]+[-1]*(K_max-len(sample[3][0])) for i in range(len(sample[3]))] + [[-1]*K_max] * (max_tlen - len(sample[3])))
        path_dir_ls.append([sample[4][i]+[[-1,-1]]*(K_max-len(sample[4][0])) for i in range(len(sample[4]))] + [[[-1,-1]]*K_max] * (max_tlen - len(sample[4])))
        path_det_ls.append([sample[5][i]+[[-1,-1,-1]]*(K_max-len(sample[5][0])) for i in range(len(sample[5]))] + [[[-1,-1,-1]]*K_max] * (max_tlen - len(sample[5])))
        path_timeslot_ls.append(sample[6] + [[-1, -1]] * (max_tlen - len(sample[6])))
        path_gridrange_ls.append(sample[7] + [[-1, -1, -1, -1, -1, -1]]* (max_tlen - len(sample[7])))
    _object = [sample[8] for sample in batch]
    _pathset = [sample[9] for sample in batch]
    _tlSeg = [sample[10] for sample in batch]
    _true_traj = [sample[11] for sample in batch]
    _true_traj_gridslist = [sample[12] for sample in batch]
    _det_area = [sample[13] for sample in batch]
    _det_name = [sample[14] for sample in batch]

    tf = torch.FloatTensor

    return tf(dist_nsp_ls), tf(dur_nsp_seconds_ls), tf(path_speed_ls), tf(dist_delta_ls),\
             tf(path_dir_ls), tf(path_det_ls), tf(path_timeslot_ls), tf(path_gridrange_ls), \
             trace_lens,ts_ks, _object, _pathset, _tlSeg, _true_traj, _true_traj_gridslist, _det_area, _det_name

