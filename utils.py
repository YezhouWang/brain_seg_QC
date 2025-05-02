import numpy as np

def dice_coefficient(pred, target, num_classes):
    dice = []
    for i in range(num_classes):
        pred_i = (pred == i).astype(np.uint8)
        target_i = (target == i).astype(np.uint8)
        intersection = np.sum(pred_i * target_i)
        union = np.sum(pred_i) + np.sum(target_i)
        d = (2. * intersection + 1e-5) / (union + 1e-5)
        dice.append(d)
    return dice
  
