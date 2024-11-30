import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import scipy.integrate as spi
import numpy as np
import os.path as osp
import pickle


class CRF(nn.Module):
    def __init__(self,
                 segE_arg,
                 pt_arg,
                 emb_transformer_dim=512,
                 device='cpu'):
        super().__init__()
        self.segE_arg = segE_arg
        self.pt_arg = pt_arg
        self.device = device

        self.liner_pre = nn.Linear(14, emb_transformer_dim, bias=False)
        self.transformer = nn.TransformerEncoderLayer(d_model=emb_transformer_dim, nhead=8)
        self.liner_post = nn.Linear(emb_transformer_dim, 1, bias=False)


    def _get_state_embedding(self,
                            path_delta_len_norm,
                            path_dir_norm,
                            path_det_norm,
                            path_timeslot_norm,
                            path_gridrange_norm,
                            batch_mask):

        src_len_dir_det = torch.cat((path_delta_len_norm.unsqueeze(dim=-1),
                                    path_dir_norm,
                                    path_det_norm,
                                    path_timeslot_norm.unsqueeze(dim=2).repeat(1, 1, batch_mask.size(2), 1), 
                                    path_gridrange_norm.unsqueeze(dim=2).repeat(1, 1, batch_mask.size(2), 1)), dim=-1)
        src_len_dir_det = src_len_dir_det.reshape(batch_mask.size(0)*batch_mask.size(1), batch_mask.size(2), 14)

        hide_len_dir_det = self.liner_pre(src_len_dir_det)
        hide_len_dir_det = self.transformer(hide_len_dir_det)
        state_embedding = self.liner_post(hide_len_dir_det).reshape(batch_mask.size(0), batch_mask.size(1), batch_mask.size(2), 1)
        state_embedding = state_embedding.squeeze()*batch_mask
        state_embedding = F.normalize(state_embedding, dim=-1, p=2)

        return state_embedding
    

    def _get_segE(self, _dist_nsp, _dur_nsp_seconds, batch_mask):
        delta_L = self.segE_arg['delta_L']
        delta_T = self.segE_arg['delta_T']
        segE_len = - _dist_nsp/delta_L * torch.log(_dist_nsp/delta_L)
        segE_dur = - _dur_nsp_seconds/delta_T * torch.log(_dur_nsp_seconds/delta_T)
        segE = segE_len+segE_dur
        segE = torch.masked_fill(segE, ~batch_mask[:,:,0], 0).unsqueeze(dim=-1)

        return segE

    def _get_speed_mu_sigma(self, path_speed, idx, batch_mask):
        speed_mu = torch.gather(input=path_speed.nan_to_num(), dim=-1, index=idx.unsqueeze(dim=-1))
        speed_sigma = torch.sqrt(torch.nansum((path_speed-speed_mu)**2,dim=-1,keepdim=True)/torch.sum(batch_mask,dim=-1,keepdim=True))

        return speed_mu, speed_sigma

    def _get_speed_smooth(self, segE, path_speed, speed_mu, speed_sigma, batch_mask):

        path_speed_smooth = path_speed.clone()
        a = self.pt_arg['a']
        eta = self.pt_arg['eta']
        alpha = self.pt_arg['alpha']

        bl = (path_speed_smooth < speed_mu - alpha*speed_sigma)*batch_mask
        up = (path_speed_smooth > speed_mu + alpha*speed_sigma)*batch_mask
        speed_abnormal_mask = (bl | up)
        if speed_abnormal_mask.sum() > 0:
            speed_mv_pad = nn.functional.pad(input=speed_mu.squeeze(-1), pad=(a, a), mode="constant", value=0)
            segE_pad = nn.functional.pad(input=segE.squeeze(-1), pad=(a, a), mode="constant", value=0)
            speed_mv_rectify = torch.zeros_like(speed_mu)
            for i_ts in range(batch_mask.size(0)):
                for i in range(batch_mask[i_ts,:,0].sum()):
                    _segE_a = segE_pad[i_ts][i: i + 2 * a + 1]
                    _v_mu_a = speed_mv_pad[i_ts][i: i + 2 * a + 1]
                    v_ab_rectify = 0
                    for j in range(1, a + 1):
                        i_v = _segE_a[a - j] / \
                            (_segE_a.sum() - _segE_a[a]) * _v_mu_a[a - j]
                        iv_ = _segE_a[a + j] / \
                            (_segE_a.sum() - _segE_a[a]) * _v_mu_a[a + j]
                        v_ab_rectify += (
                            2 * eta[j - 1] * (i_v + iv_)
                            if (i_v != 0) and (iv_ != 0)
                            else eta[j - 1] * (_v_mu_a[a - j] + _v_mu_a[a + j])
                        )
                    speed_mv_rectify[i_ts][i] = v_ab_rectify
            while True:
                path_speed_smooth[speed_abnormal_mask] = speed_mv_rectify.repeat(1,1, path_speed_smooth.size(-1))[speed_abnormal_mask]
                bl = (path_speed_smooth < speed_mu - 0*speed_sigma)*batch_mask
                up = (path_speed_smooth > speed_mu + 0*speed_sigma)*batch_mask
                smoothed = (path_speed_smooth == speed_mv_rectify)*batch_mask
                speed_abnormal_mask = (bl | up) & ~smoothed
                if speed_abnormal_mask.sum() == 0:
                    break
                
        speed_sigma_rectify = torch.sqrt(torch.nansum((path_speed_smooth-speed_mu)**2,dim=-1,keepdim=True)/torch.sum(batch_mask,dim=-1,keepdim=True))
        speed_sigma_rectify_mean = torch.masked_fill(speed_sigma_rectify, speed_sigma_rectify==0, float('nan')).nanmean(dim=1).unsqueeze(dim=-1)
        speed_sigma_rectify[speed_sigma_rectify==0] = speed_sigma_rectify_mean.repeat(1,speed_sigma_rectify.size(1),1)[speed_sigma_rectify==0]

        return path_speed_smooth, speed_mu, speed_sigma_rectify

    def _get_tran_embedding(self, segE, path_speed_smooth, speed_sigma_rectify, batch_mask):
        alpha = self.pt_arg['alpha']
        tran_embedding = torch.zeros(batch_mask.size(0),
                        batch_mask.size(1),
                        batch_mask.size(2), 
                        batch_mask.size(2)).to(self.device)

        for i_ts in range(batch_mask.size(0)):
            for i in range(1,batch_mask[i_ts,:,0].sum()):
                tv_i_list = []
                if segE[i_ts][i] < segE[i_ts][i-1]:
                    dist_list = [Normal(path_speed_smooth[i_ts][i-1][k],speed_sigma_rectify[i_ts][i-1]) for k in range(batch_mask[i_ts][i-1].sum())]

                    for k in range(batch_mask[i_ts][i-1].sum()):
                        _tv_i_ = dist_list[k].cdf(path_speed_smooth[i_ts][i][:batch_mask[i_ts][i].sum()]+alpha*speed_sigma_rectify[i_ts][i-1])-\
                                dist_list[k].cdf(path_speed_smooth[i_ts][i][:batch_mask[i_ts][i].sum()]-alpha*speed_sigma_rectify[i_ts][i-1])
                        tv_i_list.append(_tv_i_)
                    _tv_i = torch.stack(tv_i_list)

                else:
                    dist_list = [Normal(path_speed_smooth[i_ts][i][k],speed_sigma_rectify[i_ts][i]) for k in range(batch_mask[i_ts][i].sum())]

                    for k in range(batch_mask[i_ts][i].sum()):
                        _tv_i_ = dist_list[k].cdf(path_speed_smooth[i_ts][i-1][:batch_mask[i_ts][i-1].sum()]+alpha*speed_sigma_rectify[i_ts][i])-\
                                dist_list[k].cdf(path_speed_smooth[i_ts][i-1][:batch_mask[i_ts][i-1].sum()]-alpha*speed_sigma_rectify[i_ts][i])
                        tv_i_list.append(_tv_i_)
                    _tv_i = torch.stack(tv_i_list).T
                tran_embedding[i_ts][i][:_tv_i.size(0),:_tv_i.size(1)]=_tv_i
                #break
        tran_embedding = F.normalize(tran_embedding, p=2, dim=-1)
        return tran_embedding
    
    def _get_tran_embedding_one_stop(self, segE, path_speed,idx, batch_mask):
        speed_mu, speed_sigma = self._get_speed_mu_sigma(path_speed, idx, batch_mask)
        path_speed_smooth, speed_mu, speed_sigma_rectify = self._get_speed_smooth(segE, path_speed, speed_mu, speed_sigma, batch_mask)
        tran_embedding = self._get_tran_embedding(segE, path_speed_smooth, speed_sigma_rectify, batch_mask)

        return tran_embedding

    def _compute_score(self, state_embedding, tran_embedding, batch_mask):
        """
        S(X,y)
        emissions: (seq_length, batch_size, num_tags)
        seq_mask: (seq_length, batch_size)
        transitions: (seq_length, batch_size, num_tags, num_tags)
        return: (batch_size, )
        """

        emissions = state_embedding.transpose(0, 1)
        transitions = tran_embedding.transpose(0, 1)
        seq_mask = batch_mask[:,:,0].transpose(0, 1)

        seq_length, batch_size = seq_mask.shape
        
        score = torch.zeros(batch_size).to(self.device)
        score += emissions[0, torch.arange(batch_size), 0]
        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += transitions[i,:,0, 0] * seq_mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), 0] * seq_mask[i]
            
        return score

    def _compute_normalizer(self, state_embedding, tran_embedding, batch_mask):
        """
        emissions: (seq_length, batch_size, num_tags)
        seq_mask: (seq_length, batch_size)
        transitions: (seq_length, batch_size, num_tags, num_tags)
        """
        
        #emissions = torch.masked_fill(state_embedding, ~batch_mask, float('-inf')).transpose(0, 1)
        emissions = state_embedding.transpose(0, 1)
        transitions = tran_embedding.transpose(0, 1)
        seq_mask = batch_mask[:,:,0].transpose(0, 1)

        seq_length = emissions.size(0)
        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = emissions[0, :, :]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i, :, :].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + transitions[i] + broadcast_emissions
            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(seq_mask[i].unsqueeze(1), next_score, score)

        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, state_embedding, tran_embedding, batch_mask):

        """
        emissions: (seq_length, batch_size, num_tags)
        seq_mask: (seq_length, batch_size)
        trans: (seq_length, batch_size, num_tags, num_tags)
        """

        emissions = torch.masked_fill(state_embedding, ~batch_mask, float('-inf')).transpose(0, 1)
        trans = tran_embedding.transpose(0, 1)
        seq_mask = batch_mask[:,:,0].transpose(0, 1)

        seq_length, batch_size = seq_mask.shape

        # Start transition and first emission
        score = emissions[0, :, :]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        # next_score = torch.zeros(batch_size, self.num_tags).to(self.device)
        # indices = torch.zeros(batch_size, self.num_tags).int()
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, k, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, k)
            broadcast_emission = emissions[i, :, :].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, k, k)
            next_score = broadcast_score + trans[i] + broadcast_emission
            # for j in range(batch_size):
            #     cur_score, cur_indices = torch.max(score[j].unsqueeze(1) + trans + emissions[i,j,:].unsqueeze(0), dim=0)
            #     next_score[j] = cur_score
            #     indices[j] = cur_indices.detach().cpu()
            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)
            # print(next_score.shape)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(seq_mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # Now, compute the best path for each sample
        # shape: (batch_size,)
        seq_ends = seq_mask.long().sum(dim=0) - 1
        best_tags_list = []
        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]
            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            tags_len = len(best_tags)
            best_tags_list.append(best_tags + [-1] * (seq_length - tags_len))
        return best_tags_list
    
    def forward(self,
                path_delta_len_norm,
                path_dir_norm,
                path_det_norm,
                path_timeslot_norm,
                path_gridrange_norm,
                dist_nsp,
                dur_nsp_seconds,
                path_speed,
                batch_mask):

        state_embedding=self._get_state_embedding(path_delta_len_norm, path_dir_norm, path_det_norm, path_timeslot_norm, path_gridrange_norm, batch_mask)
        segE=self._get_segE(dist_nsp, dur_nsp_seconds,batch_mask)
        idx = torch.argmax(torch.masked_fill(state_embedding, ~batch_mask, float('-inf')),dim=-1)
        tran_embedding = self._get_tran_embedding_one_stop(segE,path_speed,idx,batch_mask)

        numerator = self._compute_score(state_embedding, tran_embedding, batch_mask)
        denominator = self._compute_normalizer(state_embedding, tran_embedding, batch_mask)

        llh = numerator - denominator
        seq_mask = batch_mask[:,:,0].transpose(0, 1)

        return - llh.sum() / seq_mask.float().sum()

    def infer(self,
            path_delta_len_norm,
            path_dir_norm,
            path_det_norm,
            path_timeslot_norm,
            path_gridrange_norm,
            dist_nsp,
            dur_nsp_seconds,
            path_speed,
            batch_mask):
        
        seq_mask = batch_mask[:,:,0]
        state_embedding=self._get_state_embedding(path_delta_len_norm, path_dir_norm, path_det_norm, path_timeslot_norm, path_gridrange_norm, batch_mask)
        segE=self._get_segE(dist_nsp, dur_nsp_seconds,batch_mask)
        infer_path_list = []
        cnt = 0
        itr = 0
        idx = torch.argmax(torch.masked_fill(state_embedding, ~batch_mask, float('-inf')),dim=-1)
        tran_embedding = self._get_tran_embedding_one_stop(segE,path_speed,idx,batch_mask)
        infer_path = self._viterbi_decode(state_embedding, tran_embedding,batch_mask)
        """
        while True:
            itr += 1
            tran_embedding = self._get_tran_embedding_one_stop(segE,path_speed,idx,batch_mask)
            infer_path = self._viterbi_decode(state_embedding, tran_embedding,batch_mask)
            idx = torch.masked_fill(torch.tensor(infer_path).to(self.device), ~seq_mask, 0)
            infer_path_list.append(str(infer_path))
            if cnt < len(set(infer_path_list)):
                cnt = len(set(infer_path_list))
            else:
                break
            if itr > 10:
                break
        """
        
        return infer_path