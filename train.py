import sys
import argparse
import torch.utils
from wpod import WPODNet
from dataset import LPDataset
from torch.utils.data import DataLoader
from src.loss import MyLoss
from os.path import isdir, basename
from os import makedirs

def train(train_loader, model, optimizer):
	for step, (data, target) in enumerate(train_loader):
		data = data.cuda()
		target = target.cuda()
		optimizer.zero_grad()
		prob, bbox= model(data)
		Loss = criterion(prob, bbox,target)
		Loss.backward(Loss.clone().detach())
		Loss = Loss.data
		optimizer.step()
		return Loss
if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-n' 		,'--name'			,type=str   , default='pytorch-model'		,help='Model name')
	parser.add_argument('-tr'		,'--train-dir'		,type=str   , default='data/train/'		,help='Input data directory for training')
	parser.add_argument('-its'		,'--iterations'		,type=int   , default=200000	,help='Number of mini-batch iterations (default = 300.000)')
	parser.add_argument('-bs'		,'--batch-size'		,type=int   , default=52		,help='Mini-batch size (default = 32)')
	parser.add_argument('-od'		,'--output-dir'		,type=str   , default='./weight'		,help='Output directory (default = ./)')
	parser.add_argument('-op'		,'--optimizer'		,type=str   , default='Adam'	,help='Optmizer (default = Adam)')
	parser.add_argument('-lr'		,'--learning-rate'	,type=float , default=.001		,help='Optmizer (default = 0.01)')
	parser.add_argument('--rect', action='store_true', help='rectangular training')
	parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
	args = parser.parse_args()

	netname 	= basename(args.name)
	train_dir 	= args.train_dir
	outdir 		= args.output_dir

	iterations 	= args.iterations
	batch_size 	= args.batch_size
	dim 		= 208
	if not isdir(outdir):
		makedirs(outdir)

	if not torch.cuda.is_available():
		sys.exit(1)
	print('Checking input directory...')
#	torch.cuda.set_device(args.gpu)
	dataset = LPDataset(train_dir, 208, batch_size)
	train_loader = DataLoader(dataset,batch_size = batch_size, shuffle=True)
	model = WPODNet()
	model.cuda()
	optimizer = torch.optim.Adam(model.parameters(),lr = args.learning_rate)
	criterion = MyLoss()
	model_path_backup = '%s/%s_backup' % (outdir,netname)
	model_path_final  = '%s/%s_final'  % (outdir,netname)
	min_loss = None
	for it in range(args.iterations):
		train_loss = train(train_loader, model, optimizer)
		train_loss=torch.mean(train_loss)
		print('Iter. %d (of %d)' % (it + 1, iterations))
		if min_loss is None:
			min_loss = train_loss
		else:
			if train_loss < min_loss:
				print('Saving model (%s)' % model_path_backup)
				torch.save(model.state_dict(), model_path_backup + '.pth')
				min_loss = train_loss
		print('\tLoss: {} \tBest Loss: {}'.format(train_loss, min_loss))
		if (it + 1) % 100 == 0:
			print('Saving model (%s)' % model_path_backup)
			torch.save(model.state_dict(), model_path_backup+'.pth')
	print('Over')
	print('Saving model (%s)' % model_path_final)
	torch.save(model.state_dict(), model_path_final+'.pth')

