import torch
import cv2
from wpod import WPODNet
from src.utils 					import im2single
from src.pytorch_utils 			import load_model, detect_lp
import numpy as np
from src.drawing_utils			import draw_losangle
import argparse
if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--image'	, type=str, default = './data/test/10.jpg'		,help='Input Image')
	parser.add_argument('-v', '--vtype'	, type=str, default = 'fullimage'		,help = 'Image type (car, truck, bus, bike or fullimage)')
	parser.add_argument('-t' 		,'--lp_threshold', type=float   , default = 0.45		,help = 'Detection Threshold')
	parser.add_argument('-od' ,      '--weight' ,type=str, default='./weight/wpodnet.pth' , help='Output directory (default = ./)')
	args = parser.parse_args()
	lp_threshold = args.lp_threshold
	ocr_input_size = [80, 240]  # desired LP size (width x height)
	wpod_net_path = args.weight
	model = WPODNet()
	model.load_state_dict(torch.load(wpod_net_path, map_location="cuda:0"))
	model.cuda()
	model.eval()
	image = cv2.imread(args.image)
	vtype = args.vtype
	iwh = np.array(image.shape[1::-1], dtype=float).reshape((2, 1))

	ratio = float(max(image.shape[:2])) / min(image.shape[:2])  # resizing ratio
	side = int(ratio * 288.)
	bound_dim = min(side + (side % (2 ** 4)), 608)

	if (vtype in ['car', 'bus', 'truck']):
		#  Defines crops for car, bus, truck based on input aspect ratio (see paper)
		ASPECTRATIO = max(1, min(2.75, 1.0*image.shape[1]/image.shape[0]))  # width over height
		WPODResolution = 256# faster execution
		lp_output_resolution = tuple(ocr_input_size[::-1])
	elif  vtype == 'fullimage':
		#  Defines crop if vehicles were not cropped
		ASPECTRATIO = 1
		WPODResolution = 480 # larger if full image is used directly
		lp_output_resolution =  tuple(ocr_input_size[::-1])
	else:
		#  Defines crop for motorbike
		ASPECTRATIO = 1.0 # width over height
		WPODResolution = 208
		lp_output_resolution = (int(1.5*ocr_input_size[0]), ocr_input_size[0]) # for bikes, the LP aspect ratio is lower
	#  Runs IWPOD-NET. Returns list of LP data and cropped LP images
	with torch.no_grad():
		Llp,LlpImgs = detect_lp(model,im2single(image),ASPECTRATIO*WPODResolution,2**4,(240,80),lp_threshold)
		for i, img in enumerate(LlpImgs):
		#  Draws LP quadrilateral in input image
			pts = Llp[i].pts * iwh
			draw_losangle(image, pts, color = (0,0,255.), thickness = 2)
			cv2.imshow('Rectified plate %d'%i, img )
	cv2.imshow('Image and LPs', image )
	cv2.waitKey()
	cv2.destroyAllWindows()



