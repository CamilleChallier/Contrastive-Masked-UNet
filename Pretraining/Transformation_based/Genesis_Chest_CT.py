# Original code from https://github.com/MrGiovanni/ModelsGenesis

import sys
import torch
import warnings
import random
warnings.filterwarnings('ignore')

import numpy as np
from torch import nn
from torchsummary import summary
from utils import *
from config import models_genesis_config
from tqdm import tqdm
from PIL import Image
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from Finetuning.model import UNet
from sklearn.model_selection import train_test_split

conf = models_genesis_config()
conf.display()

imagePaths = [os.path.join(conf.data, image_id) for image_id in sorted(os.listdir(conf.data))]
maskPaths = [os.path.join(conf.data, image_id) for image_id in sorted(os.listdir(conf.data))]
X_train, X_test, y_train, _ = train_test_split(imagePaths, maskPaths, test_size=0.2, random_state=42)
X_pretrain, _, _, _ = train_test_split(X_train, y_train, test_size=conf.ratio/0.8, random_state=42)

if conf.arcade == True :
    data_path = "../dataset_arcane/train/imgs"
    imagePathsArcane_train = [os.path.join(data_path, image_id) for image_id in sorted(os.listdir(data_path))]
    data_path = "../dataset_arcane/test/imgs"
    imagePathsArcane_test = [os.path.join(data_path, image_id) for image_id in sorted(os.listdir(data_path))]
    
    X_pretrain.extend(imagePathsArcane_train)
    X_test.extend(imagePathsArcane_test)

    random.shuffle(X_pretrain)
    random.shuffle(X_test)

x_train = []
for i,file_name in enumerate(tqdm(X_pretrain)):
    s = np.load(os.path.join(conf.data, file_name))
    # s = s[..., np.newaxis]
    s = Image.fromarray(s)
    s = s.resize((256, 256), resample= Image.BICUBIC)
    x_train.append(np.asarray(s))
x_train = np.array(x_train)

x_valid = []
for i,file_name in enumerate(tqdm(X_test)):
    s = np.load(os.path.join(conf.data, file_name))
    s = Image.fromarray(s)
    s = s.resize((256, 256), resample= Image.BICUBIC)
    x_valid.append(np.asarray(s))
x_valid = np.array(x_valid)

print("x_train: {} | {:.2f} ~ {:.2f}".format(x_train.shape, np.min(x_train), np.max(x_train)))
print("x_valid: {} | {:.2f} ~ {:.2f}".format(x_valid.shape, np.min(x_valid), np.max(x_valid)))

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if conf.model == "MAE" : 
	training_generator = generate_pair_mae(x_train,conf.batch_size, conf, device)
	image, gt = next(training_generator) 
	validation_generator = generate_pair_mae(x_valid,conf.batch_size, conf, device)
elif conf.model == "Model Genesis" : 
	training_generator = generate_pair(x_train,conf.batch_size, conf, device)
	image, gt = next(training_generator) 
	validation_generator = generate_pair(x_valid,conf.batch_size, conf, device)
else :
    raise ValueError(f"Unsupported method: {conf.method}. Supported methods are 'MAE' and 'Model Genesis'.")
    
model = UNet(out_classes=1)
model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
model.to(device)

print("Total CUDA devices: ", torch.cuda.device_count())

summary(model, (conf.input_rows,conf.input_cols), batch_size=-1)
criterion = nn.MSELoss()

if conf.optimizer == "sgd":
	optimizer = torch.optim.SGD(model.parameters(), 1e-2, momentum=0.9, weight_decay=0.0, nesterov=False)
elif conf.optimizer == "adam":
	optimizer = torch.optim.Adam(model.parameters(), 1e-2, conf.lr)
else:
	raise

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(conf.patience * 0.8), gamma=0.5)

# to track the training loss as the model trains
train_losses = []
# to track the validation loss as the model trains
valid_losses = []
# to track the average training loss per epoch as the model trains
avg_train_losses = []
# to track the average validation loss per epoch as the model trains
avg_valid_losses = []
best_loss = 100000
intial_epoch =0
num_epoch_no_improvement = 0
sys.stdout.flush()

if conf.weights != None:
	checkpoint=torch.load(conf.weights)
	model.load_state_dict(checkpoint['state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	intial_epoch=checkpoint['epoch']
	print("Loading weights from ",conf.weights)
sys.stdout.flush()

fold_train_losses={}
fold_valid_losses={}
for epoch in range(intial_epoch,conf.nb_epoch):
	scheduler.step(epoch)
	model.train()
	for iteration in range(int(x_train.shape[0]//conf.batch_size)):
		image, gt = next(training_generator) 
		pred=model(image)
		val_gt = gt.unique()
		values = pred.unique()
		loss = criterion(pred.squeeze(1).to(device),gt.to(device))
		optimizer.zero_grad()
		loss.backward()
  
		optimizer.step()
		train_losses.append(round(loss.item(), 2))
		if (iteration + 1) % 5 ==0:
			print('Epoch [{}/{}], iteration {}, Loss: {:.6f}'
				.format(epoch + 1, conf.nb_epoch, iteration + 1, np.average(train_losses)))
			sys.stdout.flush()
   
	with torch.no_grad():
		model.eval()
		print("validating....")
		for i in range(int(x_valid.shape[0]//conf.batch_size)):
			x,y = next(validation_generator)
			y = np.repeat(y,conf.nb_class,axis=1)
			image,gt = x.float(), y.float()
			image=image.to(device)
			gt=gt.to(device)
			pred=model(image)
			loss = criterion(pred.squeeze(1).to(device),gt.to(device))
			valid_losses.append(loss.item())

	fold_train_losses[f'fold_{epoch}'] = train_losses 
	fold_valid_losses[f'fold_{epoch}'] = valid_losses  
	
	#logging
	train_loss=np.average(train_losses)
	valid_loss=np.average(valid_losses)
	avg_train_losses.append(train_loss)
	avg_valid_losses.append(valid_loss)
	print("Epoch {}, validation loss is {:.4f}, training loss is {:.4f}".format(epoch+1,valid_loss,train_loss))
	train_losses=[]
	valid_losses=[]
	if valid_loss < best_loss:
		print("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, valid_loss))
		best_loss = valid_loss
		num_epoch_no_improvement = 0
		#save model
		torch.save({
			'epoch': epoch+1,
			'state_dict' : model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict()
		},os.path.join(conf.model_path, f"{conf.exp_name}.pt"))
		print("Saving model ",os.path.join(conf.model_path, f"{conf.exp_name}.pt"))
	else:
		print("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss,num_epoch_no_improvement))
		num_epoch_no_improvement += 1
	if num_epoch_no_improvement == conf.patience:
		print("Early Stopping")
		break
	sys.stdout.flush()

import pickle
with open(f'{conf.exp_name}_train_valid_losses.pkl', 'wb') as f:
    pickle.dump({'train_losses': fold_train_losses, 'valid_losses': fold_valid_losses}, f)

