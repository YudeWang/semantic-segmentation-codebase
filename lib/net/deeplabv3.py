# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from net.backbone import build_backbone
from net.operators import ASPP, GloRe
from utils.registry import NETS
from utils import pyutils

class _deeplabv3(nn.Module):	
	def __init__(self, cfg, batchnorm=nn.BatchNorm2d, **kwargs):
		super(_deeplabv3, self).__init__()
		self.cfg = cfg
		self.backbone = build_backbone(cfg.MODEL_BACKBONE, pretrained=cfg.MODEL_BACKBONE_PRETRAIN, **kwargs)
		self.batchnorm = batchnorm
		input_channel = self.backbone.OUTPUT_DIM	
		self.aspp = ASPP(dim_in=input_channel, 
				dim_out=cfg.MODEL_ASPP_OUTDIM, 
				rate=[0, 6, 12, 18],
				bn_mom = cfg.TRAIN_BN_MOM,
				has_global = cfg.MODEL_ASPP_HASGLOBAL,
				batchnorm = self.batchnorm)
	def __initial__(self):
		for m in self.modules():
			if m not in self.backbone.modules():
				if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
					nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				elif isinstance(m, self.batchnorm):
					nn.init.constant_(m.weight, 1)
					nn.init.constant_(m.bias, 0)
		
	def forward(self, x):
		raise NotImplementedError

@NETS.register_module
class deeplabv3(_deeplabv3):
	def __init__(self, cfg, **kwargs):
		super(deeplabv3, self).__init__(cfg, **kwargs)
		#self.dropout1 = nn.Dropout(0.5)
		self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
		self.__initial__()

	def forward(self, x):
		n,c,h,w = x.size()
		x_bottom = self.backbone(x)[-1]
		feature = self.aspp(x_bottom)
		#feature = self.dropout1(feature)
		result = self.cls_conv(feature)
		#result = F.adaptive_max_pool2d(result, 1)
		result = F.interpolate(result,(h,w),mode='bilinear', align_corners=True)
		return result

@NETS.register_module
class deeplabv3base(nn.Module):	
	def __init__(self, cfg, batchnorm=nn.BatchNorm2d, **kwargs):
		super(deeplabv3base, self).__init__()
		self.cfg = cfg
		self.backbone = build_backbone(cfg.MODEL_BACKBONE, pretrained=cfg.MODEL_BACKBONE_PRETRAIN, **kwargs)
		self.batchnorm = batchnorm
		input_channel = self.backbone.OUTPUT_DIM	
#		self.aspp = ASPP(dim_in=input_channel, 
#				dim_out=cfg.MODEL_ASPP_OUTDIM, 
#				rate=[0, 6, 12, 18],
#				bn_mom = cfg.TRAIN_BN_MOM,
#				has_global = cfg.MODEL_ASPP_HASGLOBAL,
#				batchnorm = self.batchnorm)
		self.cls_conv = nn.Conv2d(input_channel, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
	def __initial__(self):
		for m in self.modules():
			if m not in self.backbone.modules():
				if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
					nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				elif isinstance(m, self.batchnorm):
					nn.init.constant_(m.weight, 1)
					nn.init.constant_(m.bias, 0)
		
	def forward(self, x):
		n,c,h,w = x.size()
		x_bottom = self.backbone(x)[-1]
		result = self.cls_conv(x_bottom)
		result = F.interpolate(result,(h,w),mode='bilinear', align_corners=True)
		return result

@NETS.register_module
class deeplabv3seg(deeplabv3base):	
	def __init__(self, cfg, batchnorm=nn.BatchNorm2d, **kwargs):
		super(deeplabv3seg, self).__init__(cfg,batchnorm,**kwargs)
		self.aspp = ASPP(dim_in=self.backbone.OUTPUT_DIM, 
				dim_out=cfg.MODEL_ASPP_OUTDIM, 
				rate=[0, 6, 12, 18],
				bn_mom = cfg.TRAIN_BN_MOM,
				has_global = cfg.MODEL_ASPP_HASGLOBAL,
				batchnorm = self.batchnorm)
		self.seg_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
		
	def forward(self, x):
		n,c,h,w = x.size()
		x_bottom = self.backbone(x)[-1]
		cam = self.cls_conv(x_bottom)
		cam = F.interpolate(cam,(h,w),mode='bilinear', align_corners=True)
		feature = self.aspp(x_bottom)
		seg = self.seg_conv(feature)
		seg = F.interpolate(seg,(h,w),mode='bilinear', align_corners=True)
		return cam, seg

@NETS.register_module
class deeplabv3seg2(deeplabv3base):	
	def __init__(self, cfg, batchnorm=nn.BatchNorm2d, **kwargs):
		super(deeplabv3seg2, self).__init__(cfg,batchnorm,**kwargs)
		self.aspp = ASPP(dim_in=self.backbone.OUTPUT_DIM, 
				dim_out=cfg.MODEL_ASPP_OUTDIM, 
				rate=[0, 6, 12, 18],
				bn_mom = cfg.TRAIN_BN_MOM,
				has_global = cfg.MODEL_ASPP_HASGLOBAL,
				batchnorm = self.batchnorm)
		self.skip_conv = nn.Sequential(
				nn.Conv2d(256, cfg.MODEL_ASPP_OUTDIM, 1, 1, padding=0),
				self.batchnorm(cfg.MODEL_ASPP_OUTDIM),
				nn.ReLU(inplace=True)
				)
		self.seg_conv = nn.Conv2d(2*cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
		
	def forward(self, x):
		n,c,h,w = x.size()
		l1, l2, l3, x_bottom = self.backbone(x)
		cam = self.cls_conv(x_bottom)
		cam = F.interpolate(cam,(h,w),mode='bilinear', align_corners=True)
		feature = self.aspp(x_bottom)
		N,C,H,W = l1.size()
		f = F.interpolate(feature, (H,W), mode='bilinear', align_corners=True)
		seg = self.seg_conv(torch.cat([self.skip_conv(l1),f],dim=1))
		seg = F.interpolate(seg,(h,w),mode='bilinear', align_corners=True)
		return cam, seg

@NETS.register_module
class deeplabv3branch(_deeplabv3):
	def __init__(self, cfg, **kwargs):
		super(deeplabv3branch, self).__init__(cfg, **kwargs)
		self.seg_branch = nn.Sequential(
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 1, 1, padding=0),
				self.batchnorm(cfg.MODEL_ASPP_OUTDIM),
				nn.ReLU(inplace=True),
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
				)
		self.cls_branch = nn.Sequential(
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 1, 1, padding=0),
				self.batchnorm(cfg.MODEL_ASPP_OUTDIM),
				nn.ReLU(inplace=True),
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES+1, 1, 1, padding=0),
				)
		self.__initial__()

	def forward(self, x):
		n,c,h,w = x.size()
		x_bottom = self.backbone(x)[-1]
		feature = self.aspp(x_bottom)

		seg = self.seg_branch(feature)
		seg = F.interpolate(seg, (h,w), mode='bilinear', align_corners=True)
		cls = self.cls_branch(feature)
		cls = F.interpolate(cls, (h,w), mode='bilinear', align_corners=True)
		return seg, cls
		
@NETS.register_module
class deeplabv3manifold(_deeplabv3):
	def __init__(self, cfg, **kwargs):
		super(deeplabv3manifold, self).__init__(cfg, **kwargs)
		self.filter = (1-torch.eye(cfg.MODEL_NUM_CLASSES)).unsqueeze(-1).unsqueeze(-1)
		self.seg_branch = nn.Sequential(
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 1, 1, padding=0),
				self.batchnorm(cfg.MODEL_ASPP_OUTDIM),
				nn.ReLU(inplace=True),
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
				)
		self.manifold_branch = nn.Sequential(
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 1, 1, padding=0),
				self.batchnorm(cfg.MODEL_ASPP_OUTDIM),
				nn.ReLU(inplace=True),
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
				)
		self.cls_branch = nn.Sequential(
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 1, 1, padding=0),
				self.batchnorm(cfg.MODEL_ASPP_OUTDIM),
				nn.ReLU(inplace=True),
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0),
				)
		self.predefined_featuresize = int(cfg.DATA_RANDOMCROP//8)
		self.radius = 5
		self.ind_from, self.ind_to = pyutils.get_indices_of_pairs(radius=self.radius, size=(self.predefined_featuresize, self.predefined_featuresize))
		self.ind_from = torch.from_numpy(self.ind_from); self.ind_to = torch.from_numpy(self.ind_to)
		self.dis_weight = (1-torch.eye(c)).view(1,c,c)
		self.__initial__()

	def forward(self, x, to_dense=False):
		n,c,h,w = x.size()
		x_bottom = self.backbone(x)[-1]
		feature = self.aspp(x_bottom)

		seg = self.seg_branch(feature)
		seg = F.interpolate(seg, (h,w), mode='bilinear', align_corners=True)

		cls = self.cls_branch(feature)
		cls = F.interpolate(cls, (h,w), mode='bilinear', align_corners=True)

		manifold = self.manifold_branch(feature)
		manifold = F.interpolate(manifold, (h,w), mode='bilinear', align_corners=True)

		if manifold.size(2) == self.predefined_featuresize and manifold.size(3) == self.predefined_featuresize:
			ind_from = self.ind_from
			ind_to = self.ind_to
		else:
			min_edge = min(manifold.size(2), manifold.size(3))
			radius = (min_edge-1)//2 if min_edge < self.radius*2+1 else self.radius
			ind_from, ind_to = pyutils.get_indices_of_pairs(radius, (x.size(2), x.size(3)))
			ind_from = torch.from_numpy(ind_from); ind_to = torch.from_numpy(ind_to)
		manifold = manifold.view(manifold.size(0), manifold.size(1), -1).contiguous()
		ind_from = ind_from.contiguous()
		ind_to = ind_to.contiguous()
		
		ff = torch.index_select(manifold, dim=2, index=ind_from.cuda(non_blocking=True))
		ft = torch.index_select(manifold, dim=2, index=ind_to.cuda(non_blocking=True))
		aff = torch.exp(-self.dis_weight*torch.abs(ft-ff))

		if to_dense:
			aff = aff.view(-1).cpu()
			
			ind_from_exp = torch.unsqueeze(ind_from, dim=0).expand(ft.size(2), -1).contiguous().view(-1)
			indices = torch.stack([ind_from_exp, ind_to])
			indices_tp = torch.stack([ind_to, ind_from_exp])
			
			area = x.size(2)
			indices_id = torch.stack([torch.arange(0, area).long(), torch.arange(0, area).long()])
			
			aff_mat = sparse.FloatTensor(torch.cat([indices, indices_id, indices_tp], dim=1),
							torch.cat([aff, torch.ones([area]), aff])).to_dense().cuda()
			
			return seg, cls, aff_mat

		else:
			return seg, cls, aff

#	def affmat(self, seg, manifold):
#		n,c,h,w = manifold.size()
#		hw = h*w
#		manifold = manifold.view(n,d,-1).unsqueeze(-1)		# n x c x hw x 1
#		manifold_t = torch.transpose(manifold, 2, 3)		# n x c x 1 x hw
#		manifold_delta = torch.abs(manifold-manifold_t)		# n x c x hw x hw
#		dis = F.conv2d(manifold_delta, self.filter)		# n x c x hw x hw, each channel represents for class-related distance
#			
#
#		#aff_mat = torch.min(dis, dim=3, keepdim=True)
#		
#		#count = torch.sum(seg, dim=1, keepdim=True)

@NETS.register_module
class deeplabv3_noise(_deeplabv3):
	def __init__(self, cfg, **kwargs):
		super(deeplabv3_feature, self).__init__(cfg, **kwargs)
		self.branch2 = nn.Sequential(
			nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 1, bias=False),
			nn.ReLU(inplace=True))
		self.branch3 = nn.Sequential(
			nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 1, bias=False),
			nn.ReLU(inplace=True))
		self.cls_conv1 = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
		self.cls_conv2 = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
		self.cls_conv3 = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
		self.__initial__()
	def forward(self, x, inference=False):
		result = {}
		x_bottom = self.backbone(x)[-1]
		feature = self.aspp(x_bottom)
		feature = self.dropout1(feature)
		b1 = feature
		c1 = self.cls_conv1(b1)
		result['c1'] = c1
		if inference:
			b2 = self.branch2(feature)
			b2 = b1 + b2
			c2 = self.cls_conv2(b2)
			result['c2'] = c2
			b3 = self.branch3(feature)
			b3 = b2 + b3
			c3 = self.cls_conv3(b3)
			result['c3'] = c3

		return result

@NETS.register_module
class deeplabv3_feature(_deeplabv3):
	def __init__(self, cfg, **kwargs):
		super(deeplabv3_feature, self).__init__(cfg, **kwargs)
		self.__initial__()

	def forward(self, x):
		x_bottom = self.backbone(x)[-1]
		feature = self.aspp(x_bottom)
		result = self.dropout1(feature)
		result = self.cls_conv(result)

		return feature, result

@NETS.register_module
class deeplabv3_glore(_deeplabv3):
	def __init__(self, cfg, **kwargs):
		super(deeplabv3_glore, self).__init__(cfg, **kwargs)
		self.glore = GloRe(1024, cfg.MODEL_ASPP_OUTDIM, 64, cfg.MODEL_ASPP_OUTDIM)
		self.dropout1 = nn.Dropout(0.5)
		self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
		self.__initial__()

	def forward(self, x):
		n,c,h,w = x.size()
		x_bottom = self.backbone(x)[-1]
		feature = self.aspp(x_bottom)
		feature_glore = self.glore(feature)
		result = self.dropout1(feature_glore)
		result = self.cls_conv(result)
		result = F.interpolate(result, (h,w), mode='bilinear', align_corners=True)

		return result
