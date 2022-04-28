import sys, os
import torch
import cv2
import traceback
import time
from wpod import WPODNet
from glob 						import glob
from os.path 					import splitext, basename
from src.utils 					import im2single
from src.keras_utils 			import load_model, detect_lp
from src.label 					import Shape, writeShapes


def adjust_pts(pts,lroi):
	return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))


if __name__ == '__main__':

	try:
		
		input_dir  = '/ALPR_Pytorch/samples/test0115' #sys.argv[1]
		output_dir = '/ALPR_Pytorch/samples/test0115mid'

		lp_threshold = .5

		wpod_net_path = '/ALPR_Pytorch/models/my-trained-model/my-trained-model_backup.pt'  #sys.argv[2]
		model = WPODNet()
		model.load_state_dict(torch.load(wpod_net_path,map_location="cuda:0"))
		model.cuda()
		model.eval()

		imgs_paths = glob('%s/*.jpg' % input_dir)

		print('Searching for license plates using WPOD-NET')
		start = time.clock()
		for i,img_path in enumerate(imgs_paths):
			A1 = time.clock()
#			print('\t Processing %s' % img_path)

			bname = splitext(basename(img_path))[0]
#			A3 = time.clock()
			Ivehicle = cv2.imread(img_path)
#			A3 = time.clock()-A3
#			print(A3)
			ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])   #resizing ratio
			side  = int(ratio*288.)
			bound_dim = min(side + (side%(2**4)),608)
#			print("\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))
#			A1 = time.clock() - A1
#			print(A1)
			with torch.no_grad():
				Llp,LlpImgs,runtime = detect_lp(model,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold)
#			print(runtime)
#			A2 = time.clock()
			if len(LlpImgs):
				Ilp = LlpImgs[0]
				Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
				Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

				s = Shape(Llp[0].pts)

				cv2.imwrite('%s/%s_lp.png' % (output_dir,bname),Ilp*255.)
				writeShapes('%s/%s_lp.txt' % (output_dir,bname),[s])
			A2 = time.clock()-A1
			print(A2)
		end = time.clock()
		print("\ttime: %f s" % (end-start))
	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)


