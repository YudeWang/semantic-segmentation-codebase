# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import numpy as np
import random
torch.manual_seed(1) # cpu
torch.cuda.manual_seed(1) #gpu
np.random.seed(1) #numpy
random.seed(1) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
import time

from config import config_dict
from datasets.generateData import generate_dataset
from net.generateNet import generate_net
import torch.optim as optim
from net.sync_batchnorm.replicate import patch_replication_callback
from torch.utils.data import DataLoader
from utils.configuration import Configuration
from utils.finalprocess import writelog
from utils.DenseCRF import dense_crf
from utils.test_utils import single_gpu_test

cfg = Configuration(config_dict, False)

def test_net():
	period = 'val'
	dataset = generate_dataset(cfg, period=period, transform='none')
	def worker_init_fn(worker_id):
		np.random.seed(1 + worker_id)
	dataloader = DataLoader(dataset, 
				batch_size=1, 
				shuffle=False, 
				num_workers=cfg.DATA_WORKERS,
				worker_init_fn = worker_init_fn)
	
	net = generate_net(cfg, batchnorm=nn.BatchNorm2d, dilated=cfg.MODEL_BACKBONE_DILATED, multi_grid=cfg.MODEL_BACKBONE_MULTIGRID, deep_base=cfg.MODEL_BACKBONE_DEEPBASE)
	print('net initialize')
	if cfg.TEST_CKPT is None:
		raise ValueError('test.py: cfg.MODEL_CKPT can not be empty in test period')
	
	print('start loading model %s'%cfg.TEST_CKPT)
	model_dict = torch.load(cfg.TEST_CKPT)
	net.load_state_dict(model_dict,strict=False)

	print('Use %d GPU'%cfg.GPUS)
	assert torch.cuda.device_count() == cfg.GPUS
	device = torch.device('cuda')
	net.to(device)
	net.eval()

	def prepare_func(sample):	
		image_msf = []
		for rate in cfg.TEST_MULTISCALE:
			inputs_batched = sample['image_%f'%rate]
			image_msf.append(inputs_batched)
			if cfg.TEST_FLIP:
				image_msf.append(torch.flip(inputs_batched,[3]))
		return image_msf

	def inference_func(model, img):
		cam = model(img)
		return cam

	def collect_func(result_list, sample):
		[batch, channel, height, width] = sample['image'].size()
		for i in range(len(result_list)):
			result = F.interpolate(result_list[i], (height, width), mode='bilinear', align_corners=True)	
			if cfg.TEST_FLIP and i % 2 == 1:
				result = torch.flip(result, [3])
			result_list[i] = result
		prob = torch.cat(result_list, dim=0)
		prob = torch.mean(prob, dim=0, keepdim=True)
		prob = F.softmax(prob, dim=1)

		if cfg.TEST_CRF:
			prob = prob.detach().cpu().numpy()		
			inputs = sample['image'][0].numpy()
			mean = (0.485, 0.456, 0.406)
			std = (0.229, 0.224, 0.225)
			inputs[0,:,:] = (inputs[0,:,:]*std[0] + mean[0])*255
			inputs[1,:,:] = (inputs[1,:,:]*std[1] + mean[1])*255
			inputs[2,:,:] = (inputs[2,:,:]*std[2] + mean[2])*255
			inputs[inputs > 255] = 255
			inputs[inputs < 0] = 0
			img_batched = inputs.astype(np.uint8)
			prob[0] = dense_crf(prob[0], img_batched, n_classes=cfg.MODEL_NUM_CLASSES, n_iters=1)
			prob = torch.from_numpy(prob.astype(np.float32))
		result = torch.argmax(prob, dim=1).cpu().numpy().astype(np.uint8)
		result_item = result[0,:,:]
		return result_item

	def save_step_func(result_sample):
		dataset.save_result([result_sample], cfg.MODEL_NAME)

	result_list = single_gpu_test(net, dataloader, prepare_func=prepare_func, inference_func=inference_func, collect_func=collect_func, save_step_func=save_step_func)
	resultlog = dataset.do_python_eval(cfg.MODEL_NAME)
	print('Test finished')
	writelog(cfg, period, metric=resultlog)

if __name__ == '__main__':
	test_net()


