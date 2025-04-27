import os
import shutil
from pathlib import Path

class models_genesis_config:
    model = "Model Genesis" # MAE or Model Genesis
    suffix = "genesis_chest_ct"
    exp_name = model + "-" + suffix
    
    arcade = False # Use arcade dataset for pretraining. Dataset in ../dataset_arcane/train/imgs and test/imgs
    ratio = 0.1 # ratio initial dataset
    
    # data
    current_path = Path(__file__).resolve()
    data = current_path.parents[2] / "dataset" / "imgs"

    input_rows = 64
    input_cols = 64 
    input_deps = 32
    nb_class = 1
    
    # model pre-training
    verbose = 1
    weights = None
    batch_size = 64
    optimizer = "sgd"
    workers = 10
    max_queue_size = workers * 4
    save_samples = "png"
    nb_epoch = 256
    patience = 50
    lr = 1

    # image deformation
    nonlinear_rate = 0.9
    paint_rate = 0.9
    outpaint_rate = 0.8
    inpaint_rate = 1.0 - outpaint_rate
    local_rate = 0.5
    flip_rate = 0.4
    
    # logs
    model_path = "/data/challier/LTS4_coronary_segmentation/selfsupervised_pretraining/Contrastive_learning/pretrained_weights"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    logs_path = os.path.join(model_path, "Logs")
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
