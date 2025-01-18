import torch
import torch.nn as nn
import torch.nn.functional as F

import model.resnet as models
import model.vgg as vgg_models
from model.PFENet import *
import pdb

def get_correlation_mask(supp_feat, query_feat, supp_mask):
    cosine_eps = 1e-7
    
    # Resize the support mask to match the feature size
    resize_size = supp_feat.size(2)
    supp_mask_resized = F.interpolate(supp_mask, size=(resize_size, resize_size), mode='bilinear', align_corners=True)
    
    # Apply the mask to the support feature
    supp_feat_4 = supp_feat * supp_mask_resized
    
    # Query feature and support feature normalization
    q = query_feat
    s = supp_feat_4
    bsize, ch_sz, sp_sz, _ = q.size()[:]

    # Flatten and normalize query and support features
    tmp_query = q.contiguous().view(bsize, ch_sz, -1)
    tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

    tmp_supp = s               
    tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1) 
    tmp_supp = tmp_supp.contiguous().permute(0, 2, 1) 
    tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True) 


    # Compute cosine similarity
    similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
    similarity = similarity.max(1)[0].view(bsize, sp_sz * sp_sz)

    # Normalize the similarity
    similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
    
    # Reshape and interpolate the similarity map to the query feature map size
    corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)
    corr_query = F.interpolate(corr_query, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True)
    
    return corr_query


def extract_features(pfenet, x, s_x, s_y, us_x):
    """
    提取中层、高层特征和先验掩码信息。

    参数:
        pfenet: PFENet 模型实例
        x: 查询图像
        s_x: labeled支持图像
        s_y: 标签
        us_x: Unlabeled支持图像

    返回:
        F_supp_mid: 支持图像的中层特征
        F_query_mid: 查询图像的中层特征
        Prior_mask_supp: 支持图像的先验掩码
        Prior_mask_query: 查询图像的先验掩码
        F_supp_high: 支持图像的高层特征
        F_query_high: 查询图像的高层特征
    """
    # 提取查询图像的特征
    # pdb.set_trace()
    with torch.no_grad():
        # 查询图像的前向传播
        query_feat_0 = pfenet.layer0(x)
        query_feat_1 = pfenet.layer1(query_feat_0)
        query_feat_2 = pfenet.layer2(query_feat_1)
        query_feat_3 = pfenet.layer3(query_feat_2)
        query_feat_4 = pfenet.layer4(query_feat_3)

    F_query_mid = torch.cat([query_feat_3, query_feat_2], 1)
    F_query_high = query_feat_4

    # 提取第一个样本的高层特征和掩码信息
    mask = (s_y[:, 0, :, :] == 1).float().unsqueeze(1)
    with torch.no_grad():
        # 支持图像的前向传播
        supp_feat_0 = pfenet.layer0(s_x[:, 0, :, :, :])
        supp_feat_1 = pfenet.layer1(supp_feat_0)
        supp_feat_2 = pfenet.layer2(supp_feat_1)
        supp_feat_3 = pfenet.layer3(supp_feat_2)
        mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)
        supp_feat_4 = pfenet.layer4(supp_feat_3 * mask)

    # 提取unlabeled support图像的特征
    with torch.no_grad():
        # 查询图像的前向传播
        us_feat_0 = pfenet.layer0(us_x[:, 0, :, :, :])
        us_feat_1 = pfenet.layer1(us_feat_0)
        us_feat_2 = pfenet.layer2(us_feat_1)
        us_feat_3 = pfenet.layer3(us_feat_2)
        us_feat_4 = pfenet.layer4(us_feat_3)

    F_supp_mid = torch.cat([us_feat_3, us_feat_2], 1)
    F_supp_high = us_feat_4

    # 先验掩码获取
    Prior_mask_supp = get_correlation_mask(supp_feat_4, F_supp_high, mask)
    Prior_mask_query = get_correlation_mask(supp_feat_4, F_query_high, mask)
    return F_supp_mid, F_query_mid, Prior_mask_supp, Prior_mask_query, F_supp_high, F_query_high


class ActiveLearningModule(nn.Module):
    def __init__(self, mid_channels, high_channels):
        super(ActiveLearningModule, self).__init__()
        
        # 中层特征卷积
        self.conv_mid = nn.Sequential(
            nn.Conv2d(2 * mid_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 进一步压缩通道
            nn.ReLU()
        )
        
        # 先验掩码扩展和卷积
        self.expand_prior = nn.Conv2d(2, 64, kernel_size=1)  # 将通道数从 2 扩展到 64
        self.conv_prior = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # 进一步压缩通道
            nn.ReLU()
        )
        
        # 高层特征卷积
        self.conv_high = nn.Sequential(
            nn.Conv2d(2 * high_channels, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 进一步压缩通道
            nn.ReLU()
        )
        
        # 多尺度特征融合
        self.multi_scale_fusion = nn.Sequential(
            nn.Conv2d(64 + 16 + 128, 256, kernel_size=3, padding=1),  # 输入通道数为 64 + 16 + 128
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),  # 输出通道数为 1
            nn.Sigmoid()  # 注意力权重归一化到 [0, 1]
        )
        
        # 最终卷积层，输出通道数为 1
        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)  # 输出通道数为 1
        )
        self.sigmoid = nn.Sigmoid()  # 用于得分归一化

    def forward(self, F_supp_mid, F_query_mid, Prior_mask_supp, Prior_mask_query, F_supp_high, F_query_high):
        # 分别拼接
        F_mid_concat = torch.cat([F_supp_mid, F_query_mid], dim=1)  # [2 * C1, H, W]
        F_prior_concat = torch.cat([Prior_mask_supp, Prior_mask_query], dim=1)  # [2, H, W]
        F_high_concat = torch.cat([F_supp_high, F_query_high], dim=1)  # [2 * C2, H, W]
        
        # 分别卷积
        F_mid = self.conv_mid(F_mid_concat)  # [64, H, W]
        F_prior_expanded = self.expand_prior(F_prior_concat)  # [64, H, W]
        F_prior = self.conv_prior(F_prior_expanded)  # [16, H, W]
        F_high = self.conv_high(F_high_concat)  # [128, H, W]
        
        # 多尺度特征融合
        F_final_concat = torch.cat([F_mid, F_prior, F_high], dim=1)  # [64 + 16 + 128, H, W]
        F_fused = self.multi_scale_fusion(F_final_concat)  # [128, H, W]
        
        # 注意力机制
        attention_weights = self.attention(F_fused)  # [1, H, W]
        F_attended = F_fused * attention_weights  # 应用注意力权重
        
        # 最终卷积层
        score = self.final_conv(F_attended)  # [batch_size, 1, H, W]
        score = self.sigmoid(score)  # 归一化得分到 [0, 1]
        score = score.squeeze(1)  # [batch_size, H, W]
        
        return score

    # 计算损失差异
    def calculate_loss_difference(self, pfenet, x, s_x1, s_y1, candidate_s_x, candidate_s_y, y):
        # PFENet的前向传播，使用第一个支持样本 s_x1 和 s_y1
        out, main_loss, aux_loss = pfenet(x, s_x1, s_y1, y)

        loss_diffs = []  # 用来存储每个候选支持样本的损失差异
        # 遍历候选支持样本集
        for i in range(candidate_s_x.size(0)):
            s_x2, s_y2 = candidate_s_x[i], candidate_s_y[i]

            # ActiveLearningModule的前向传播，使用第二个支持样本 s_x2 和 s_y2
            F_supp_mid, F_query_mid, Prior_mask_supp, Prior_mask_query, F_supp_high, F_query_high = extract_features(pfenet, x, s_x1, s_y1, s_x2)
            score = self(F_supp_mid, F_query_mid, Prior_mask_supp, Prior_mask_query, F_supp_high, F_query_high)

            # 计算ActiveLearningModule的损失
            active_learning_loss = F.mse_loss(score, y.float().unsqueeze(1))  # 使用MSE损失来比较score与y的差异

            # 计算损失差异
            loss_diff = torch.abs(active_learning_loss - main_loss)
            loss_diffs.append((loss_diff, i))  # 保存损失差异和样本索引

        return loss_diffs

    # 评估候选支持样本
    def evaluate_candidates(self, pfenet, x, s_x1, s_y1, candidate_s_x, candidate_s_y, y):
        # 计算所有候选支持样本的损失差异
        loss_diffs = self.calculate_loss_difference(pfenet, x, s_x1, s_y1, candidate_s_x, candidate_s_y, y)

        # 按损失差异排序，选择排名前 shot-1 的支持样本
        loss_diffs.sort(key=lambda x: x[0])  # 按损失差异升序排序

        # 选择排名前 shot-1 个最小损失差异的候选支持样本
        best_support_samples = loss_diffs[:self.shot-1]

        # 返回排名前 shot-1 的候选支持样本对（损失差异，索引）
        return best_support_samples


# # 假设你有查询图像 x 和标签 y，以及支持样本 x1, y1 和候选支持集 candidate_s_x, candidate_s_y
# shot = 5  # 选择前 4 个候选支持样本

# best_support_samples = active_learning_module.evaluate_candidates(
#     pfenet, x, s_x1, s_y1, candidate_s_x, candidate_s_y, y, shot
# )

# # best_support_samples 是一个包含前 shot-1 个损失差异最小的候选支持样本的列表
# # 格式：[ (loss_diff, idx1), (loss_diff, idx2), ... ]
