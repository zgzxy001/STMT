import torch
import math

import torch.nn as nn
import torch.nn.functional as F

from utils import group_points_4DV_T_S, group_points_4DV_T_S2
from channelattention import ChannelAttention, ChannelAttention0
from positionencoding import get_positional_encoding

from model.geodesic_model import GeodesicModel as geodesic
from model.spatial_model import SpatialModel as spadist

from model.transformer import Transformer


class STMT(nn.Module):
    def __init__(self, opt, num_clusters=8, gost=1, dim=128,
                 normalize_input=True):
        super(STMT, self).__init__()
        self.temperal_num = opt.temperal_num
        self.knn_K = opt.knn_K
        self.ball_radius2 = opt.ball_radius2
        self.sample_num_level1 = opt.sample_num_level1
        self.sample_num_level2 = opt.sample_num_level2
        self.INPUT_FEATURE_NUM = opt.INPUT_FEATURE_NUM  # x,y,x,c : 4
        self.framenum = opt.framenum
        self.num_outputs = opt.Num_Class
        ####SAMPLE_NUM
        self.Seg_size = opt.Seg_size
        self.stride = opt.stride
        self.EACH_FRAME_SAMPLE_NUM = opt.EACH_FRAME_SAMPLE_NUM


        self.geo = geodesic()
        self.spa = spadist()
        self.ca_T1 = ChannelAttention(
            1024)

        self.transformer = Transformer(dim=1024, depth=5, heads=8, dim_head=128, mlp_dim=512)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, self.num_outputs),
        )




    def forward(self, points4DV_T,geodist):
        b, t, point_num, feature_num = points4DV_T.size()
        # print('points4DV_T = ',
        #       points4DV_T.size())  # points4DV_T =  torch.Size([8, 24, 512, 3])
        points4DV_T_pctinput = points4DV_T.view(b * t, point_num, feature_num)
        points4DV_T_pctinput = points4DV_T_pctinput.permute(0, 2, 1)


        _,pct_logits_last_geo,new_xyz_return_1, new_xyz_return_2, new_feature_return_1, new_feature_return_2 = self.geo(points4DV_T_pctinput,geodist) # (B*f)*feature_num (8*24)*40

        _,pct_logits_last_spa = self.spa(points4DV_T_pctinput,new_xyz_return_1, new_xyz_return_2, new_feature_return_1, new_feature_return_2)

        pct_logits_last = torch.cat((pct_logits_last_geo, pct_logits_last_spa), 1)

        xt = pct_logits_last.unsqueeze(-1)

        T_sample_num_level2_pct = pct_logits_last.size()[2]

        xt = xt.view(-1, self.framenum, xt.size(1),
                     T_sample_num_level2_pct).transpose(2, 3)

        T_inputs_level2 = xt.transpose(1, 3).transpose(2, 3)

        xt = self.ca_T1(T_inputs_level2) * T_inputs_level2

        b_curr, t_curr, feature_num_1, cen_num_1 = xt.size()[0], xt.size()[2], xt.size()[1], xt.size()[3]

        p4d_input = xt.permute(0,2,3,1)

        p4d_input = p4d_input.reshape(b_curr, t*cen_num_1, feature_num_1)


        output = self.transformer(p4d_input)

        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]

        output = self.mlp_head(output)

        return output

