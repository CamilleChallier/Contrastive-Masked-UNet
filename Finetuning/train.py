import torch, sys
import argparse
import pickle
import sys
import torch
import warnings
import random
import pandas as pd
import time  

warnings.filterwarnings("ignore")
import numpy as np
from tqdm import tqdm as tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold

from dataset import SegmentationDataset, prepare_train_test,  get_training_augmentation
from model import UNet
from metrics import hausdorff, radius_arteries, soft_cldice, DiceLoss, IoU, CrossEntropyLoss
from utils import find_best_epochs


class Meter(object):
    """Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    """

    def reset(self):
        """Reset the meter to default settings."""
        pass

    def add(self, value):
        """Log a new value to the meter
        Args:
            value: Next result to include.
        """
        pass

    def value(self):
        """Get the value of the meter in the current state."""
        pass

class AverageValueMeter(Meter):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))
            
    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan

class Epoch:
    def __init__(self, model, loss, metrics, stage_name, device="cpu", verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):
        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {
            metric.__name__: AverageValueMeter() for metric in self.metrics
        }

        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    # print(metric_fn)
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics, optimizer, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        return loss, prediction


class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="valid",
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction


def train(model, train_loader, test_loader, train_epoch, test_epoch, TRAINING, EPOCHS, name = './work_dir/best_model.pth'):
    if TRAINING:

        best_dice_score = 1000
        train_logs_list, valid_logs_list = [], []

        for i in range(0, EPOCHS):

            # Perform training & validation
            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)
            valid_logs = test_epoch.run(test_loader)
            train_logs_list.append(train_logs)
            valid_logs_list.append(valid_logs)
            print("valid_logs : ", valid_logs)

            # Save model if a better dice score is obtained
            if best_dice_score > valid_logs['dice_loss']:
                best_dice_score = valid_logs['dice_loss']
                torch.save(model, name)
                print('Model saved!')
    return train_logs_list, valid_logs_list
                
def eval(test_dataloader, model, loss, metrics, DEVICE):
    test_epoch = ValidEpoch(
        model,
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )

    valid_logs = test_epoch.run(test_dataloader)
    print("Evaluation on Test Data: ")
    return valid_logs
    
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', dest = "epochs", metavar='E', type=list, default=[2], help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=list, default=[16,32], help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=list, default=[0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                        help='Learning rate', dest='lr')
    parser.add_argument('--pretrained', '-p', dest='pretrained', type=str, default=None, help='Path to a pretrained model')
    parser.add_argument('--name', '-n', dest='name', type=str, default="base", help='name of the trained model to save')
    parser.add_argument('--ratio', '-r', dest='ratio', type=float, default=0.1, help='Ratio of finetuning dataset')
    return parser.parse_args()

def load_model(args):
    
    model = UNet()
                           
    if args.pretrained != None :
        weight_dir = args.pretrained
        checkpoint = torch.load(weight_dir, map_location="cuda:1") 
            
        if args.pretrained.endswith(".pth"):
            
            if "module" in checkpoint.keys() :
                print("encoder + decoder")
                state_dict =checkpoint["module"]
                unParalled_state_dict = {}
                for key in state_dict.keys():
                    unParalled_state_dict[key.replace("sparse_encoder.sp_cnn.", "")] = state_dict[key]
                    unParalled_state_dict[key.replace("dense_decoder.", "")] = state_dict[key]
                encoder_state_dict =  {k: v for k, v in unParalled_state_dict.items() if 'down_conv' in k or "double_conv" in k or "up_conv" in k}
                encoder_state_dict.pop('conv_last.weight', None)
                encoder_state_dict.pop('conv_last.bias', None)
                model.load_state_dict(encoder_state_dict, strict=False)
            
            elif "mmengine_version" in checkpoint["meta"].keys() :
                print("CMAE")
                state_dict = checkpoint['state_dict']
                unParalled_state_dict = {}
                for key in state_dict.keys():
                    if "pixel_decoder" in key :
                        unParalled_state_dict[key.replace("pixel_decoder.", "")] = state_dict[key]
                    if "backbone" in key : 
                        unParalled_state_dict[key.replace("backbone.", "")] = state_dict[key]
                unParalled_state_dict.pop('conv_last.weight', None)
                unParalled_state_dict.pop('conv_last.bias', None)
                model.load_state_dict(unParalled_state_dict, strict=False)
            
            else : 
                print("encoder only")
                state_dict =checkpoint
                unParalled_state_dict = {}
                for key in state_dict.keys():
                    unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
                    
                encoder_state_dict = {k: v for k, v in unParalled_state_dict.items() if 'down_conv' in k or "double_conv" in k}
                encoder_state_dict.pop('conv_last.weight', None)
                encoder_state_dict.pop('conv_last.bias', None)
                model.load_state_dict(encoder_state_dict, strict=False)
        elif args.pretrained.endswith(".ckpt"):
            print("MOCO")
            state_dict = checkpoint['state_dict']
            unParalled_state_dict = {}
            for key in state_dict.keys():
                unParalled_state_dict[key.replace("encoder_q.", "")] = state_dict[key]
                
            encoder_state_dict = {k: v for k, v in unParalled_state_dict.items() if 'down_conv' in k or "double_conv" in k}
            encoder_state_dict.pop('conv_last.weight', None)
            encoder_state_dict.pop('conv_last.bias', None)
            model.load_state_dict(encoder_state_dict, strict=False)
        else : 
            print("pretrained pt")
            state_dict = checkpoint['state_dict']
            unParalled_state_dict = {}
            for key in state_dict.keys():
                unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
            unParalled_state_dict.pop('conv_last.weight', None)# Remove the pre-trained weights for the last layer to avoid size mismatch
            unParalled_state_dict.pop('conv_last.bias', None)
            model.load_state_dict(unParalled_state_dict, strict=False)
    else :
        print("random weight init")
    return model
    

def main_finetuning(args, loss, metrics, DEVICE, select_class_values, X_finetuning, y_finetuning):
    
    LRs = args.lr
    EPOCHs= args.epochs
    BATCHs= args.batch_size
    
    result = []
    score = []
    
    for LR in LRs :
        for EPOCH in EPOCHs :
            for BATCH in BATCHs : 

                model = load_model(args)
                
                kf = KFold(n_splits=3, shuffle=True, random_state=42)

                cv_results = []

                for fold, (train_idx, val_idx) in enumerate(kf.split(X_finetuning)):  
                    print(f"Fold {fold + 1}/{3}")
                    print(train_idx)
                    name = f'./work_dir/{args.name}_{LR}_{BATCH}_{fold+1}.pth'

                    # Split data into train and validation sets
                    X_ft, X_val = [X_finetuning[i] for i in train_idx], [X_finetuning[i] for i in val_idx]
                    y_ft, y_val = [y_finetuning[i] for i in train_idx], [y_finetuning[i] for i in val_idx]
                
                    train_dataset = SegmentationDataset(X_ft, y_ft, class_values=select_class_values, augmentation = get_training_augmentation())
                    test_dataset = SegmentationDataset(X_val, y_val, class_values=select_class_values)

                    optimizer = torch.optim.Adam([ 
                        dict(params=model.parameters(), lr=LR),
                    ])
                    
                    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=12)

                    train_epoch = TrainEpoch(
                        model, 
                        loss=loss, 
                        metrics=metrics, 
                        optimizer=optimizer,
                        device=DEVICE,
                        verbose=True,
                    )

                    test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False, num_workers=4)

                    test_epoch = ValidEpoch(
                        model,
                        loss=loss,
                        metrics=metrics,
                        device=DEVICE,
                        verbose=True,
                    )
                    start_time = time.time()
                    train_logs_list, valid_logs_list = train(model, train_loader, test_loader, train_epoch, test_epoch, True, EPOCH, name)
                    runtime = time.time() - start_time
                    
                    cv_results.append(find_best_epochs(valid_logs_list, EPOCH, LR, BATCH, runtime)["dice_loss"])
                    
                    result.append({"epochs": EPOCH, "lr": LR, "batch_size": BATCH, "runtime": runtime, "train_logs_list":train_logs_list,"valid_logs_list":valid_logs_list})
                
                score.append({"epochs": EPOCH, "lr": LR, "batch_size": BATCH, "dice_loss": np.mean(cv_results)})
                
    with open(f'work_dir/results_{args.name}.pkl', 'wb') as f:
        print("saved")
        pickle.dump(result, f) 
    return [min(score, key=lambda x: x["dice_loss"])[key] for key in ["lr", "batch_size", "epochs"]]

def test(args, LR, BATCH, EPOCH, loss, metrics, DEVICE, select_class_values, X_finetuning, y_finetuning):
    
    name = f'./work_dir/{args.name}.pth'

    model = load_model(args)
    
    train_dataset = SegmentationDataset(X_finetuning, y_finetuning, class_values=select_class_values, augmentation = get_training_augmentation())
    test_dataset = SegmentationDataset(X_test, y_test, class_values=select_class_values)

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=LR),
    ])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=12)

    train_epoch = TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False, num_workers=4)

    test_epoch = ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )
    start_time = time.time()
    train_logs_list, valid_logs_list = train(model, train_loader, test_loader, train_epoch, test_epoch, True, EPOCH, name)
    runtime = time.time() - start_time

    best_model = torch.load(name, map_location=DEVICE)
    print('Loaded UNet model from this run.')
    
    valid_logs = eval(test_loader, best_model, loss, metrics, DEVICE)
    
    result = {"epochs": EPOCH, "lr": LR, "batch_size": BATCH, "runtime": runtime, "train_logs_list":train_logs_list,"valid_logs_list":valid_logs_list , "valid_logs":valid_logs}

    with open(f'./work_dir/result_test_{args.name}.pkl', 'wb') as f:
        print("saved")
        pickle.dump(result, f)


if __name__ == '__main__':
    
    #set seed : 
    seed = 42

    import random
    random.seed(seed)

    import numpy as np
    np.random.seed(seed)

    import torch
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)

    args = get_args()

    TRAINING = True
    X, y = prepare_train_test()
    select_class_values =  np.array([[0],[1]])

    # Set device: `cuda` or `cpu`
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("DEVICE:",DEVICE)

    # define loss function
    loss = DiceLoss(activation =  "softmax", threshold=0.5, ignore_channels=[0]) + CrossEntropyLoss()

    # define metrics
    metrics = [
        DiceLoss(activation =  "softmax", threshold=0.5, ignore_channels=[0]),
        CrossEntropyLoss(),
        IoU(threshold=0.5, activation="softmax", ignore_channels=[0]),
        hausdorff(threshold = 0.5, activation="softmax", ignore_channels=[0]),
        radius_arteries(), 
        soft_cldice(threshold=0.5, activation="softmax", ignore_channels=[0])
    ]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    _, X_finetuning, _, y_finetuning = train_test_split(X_train, y_train, test_size=args.ratio/0.8, random_state=42)
    
    best_LR, best_BATCH, best_EPOCH = main_finetuning(args, loss, metrics, DEVICE, select_class_values, X_finetuning, y_finetuning)
    print(best_LR, best_BATCH, best_EPOCH)
    test(args, best_LR, best_BATCH, best_EPOCH, loss, metrics, DEVICE, select_class_values, X_finetuning, y_finetuning)

