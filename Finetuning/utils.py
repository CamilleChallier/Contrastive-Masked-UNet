import numpy as np


def find_best_epochs(valid_logs, epochs, lr, batch_size, runtime, metric = 'dice_loss + cross_entropy_loss'):
    """
    Finds the best epoch based on the given evaluation metric and returns a summary of results.

    Args:
        valid_logs (list of dicts): A list of dictionaries where each dictionary contains validation metrics per epoch.
        epochs (int): The total number of epochs trained.
        lr (float): The learning rate used in training.
        batch_size (int): The batch size used during training.
        runtime (float): The total runtime of the training process (in seconds or minutes).
        metric (str, optional): The key in `valid_logs` to determine the best epoch. Defaults to 'dice_loss + cross_entropy_loss'.

    Returns:
        dict: A dictionary containing:
            - "epochs" (int): The total number of epochs.
            - "lr" (float): The learning rate used.
            - "batch_size" (int): The batch size used.
            - "runtime" (float): The total training time.
            - Selected metric value from the best epoch.
            - "dice_loss" (float): Dice loss from the best epoch.
            - "cross_entropy_loss" (float): Cross-entropy loss from the best epoch.
            - "soft_clDice" (float): Soft clDice loss from the best epoch.
            - "iou_loss" (float): IoU loss from the best epoch.
            - "hausdorff" (float or None): Hausdorff distance from the best epoch. If it's `np.inf` or `NaN`, a valid previous epoch's Hausdorff value is used.
            - "radius_arteries" (float): Radius of arteries from the best epoch.
    """
    best_valid_metric =valid_logs[0][metric]
    for i, r in enumerate(valid_logs):
        current_valid_metric = r[metric] 
        if current_valid_metric < best_valid_metric:
            best_valid_metric = current_valid_metric
            best_result = i
    if valid_logs[best_result]['hausdorff'] == np.inf or np.isnan(valid_logs[best_result]['hausdorff']):
        haus = None
        # Iterate backwards from `best_result - 1` to 0
        for i in range(best_result - 1, -1, -1):
            if valid_logs[i]['hausdorff'] != np.inf and not np.isnan(valid_logs[i]['hausdorff']):
                haus = valid_logs[i]['hausdorff']
                break
        
    else : 
        haus = valid_logs[best_result]['hausdorff']

    res = {
        "epochs": epochs,
        "lr": lr,  
        "batch_size": batch_size, 
        "runtime": runtime,
        metric: valid_logs[best_result][metric],
        'dice_loss': valid_logs[best_result]['dice_loss'],
        'cross_entropy_loss': valid_logs[best_result]['cross_entropy_loss'],
        'soft_clDice': valid_logs[best_result]['soft_clDice'],
        'iou_loss': valid_logs[best_result]['iou_loss'],
        'hausdorff': haus,
        'radius_arteries': valid_logs[best_result]['radius_arteries']
    }

    return res