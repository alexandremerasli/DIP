import torch
import torch.nn

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'




def f():
		#torch.cuda.set_per_process_memory_fraction(0.01)
	print('{}MB allocated'.format(torch.cuda.memory_allocated()/1024**2))
	print(torch.cuda.memory_summary())
	torch.cuda._lazy_init()
	x = torch.randn(1,3,256,256).cuda()
	
	print('{}MB allocated'.format(torch.cuda.memory_allocated()/1024**2))
	print(torch.cuda.memory_summary())
	n = torch.nn.Conv2d(3,3,3,1,1).cuda()
	print('{}MB allocated'.format(torch.cuda.memory_allocated()/1024**2))
	print(torch.cuda.memory_summary())
	y = n(x)
	print('{}MB allocated'.format(torch.cuda.memory_allocated()/1024**2))
	print(torch.cuda.memory_summary())
	y = n(x)
	print(x,y)

f()


