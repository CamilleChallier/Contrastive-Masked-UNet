from __future__ import print_function
import math
import os
import random
import copy
import scipy
import imageio
import string
import numpy as np
from skimage.transform import resize
try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb
import torch

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])
    
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

def data_augmentation(x, y, prob=0.5):
    # augmentation by flipping
    cnt = 3
    while random.random() < prob and cnt > 0:
        degree = random.choice([0, 1])
        x = np.flip(x, axis=degree)
        y = np.flip(y, axis=degree)
        cnt = cnt - 1

    return x, y

def nonlinear_transformation(x, prob=0.5):
    if random.random() >= prob:
        return x
    min = np.min(x)
    max = np.max(x)
    points = [[min, min], [random.random() * (max - min) + min, random.random()* (max - min) + min], [random.random()* (max - min) + min, random.random()* (max - min) + min], [max,max]]
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x

def local_pixel_shuffling(x, prob=0.5):
    if random.random() >= prob:
        return x
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    img_rows, img_cols = x.shape
    num_block = 10000
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows//25) # modif 10
        block_noise_size_y = random.randint(1, img_cols//25)
        # block_noise_size_z = random.randint(1, img_deps//10)
        noise_x = random.randint(0, img_rows-block_noise_size_x)
        noise_y = random.randint(0, img_cols-block_noise_size_y)
        # noise_z = random.randint(0, img_deps-block_noise_size_z)
        window = orig_image[ noise_x:noise_x+block_noise_size_x, 
                               noise_y:noise_y+block_noise_size_y
                            #    noise_z:noise_z+block_noise_size_z,
                           ]
        window = window.flatten()
        np.random.shuffle(window)
        window = window.reshape((block_noise_size_x, 
                                 block_noise_size_y
                                #  block_noise_size_z
                                ))
        image_temp[ noise_x:noise_x+block_noise_size_x, 
                      noise_y:noise_y+block_noise_size_y
                    #   noise_z:noise_z+block_noise_size_z
                      ] = window
    local_shuffling_x = image_temp

    return local_shuffling_x

def image_in_painting(x):
    img_rows, img_cols = x.shape
    cnt = 5
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = random.randint(img_rows//6, img_rows//3) # modif 6 3
        block_noise_size_y = random.randint(img_cols//6, img_cols//3)
        # block_noise_size_z = random.randint(img_deps//6, img_deps//3)
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        # noise_z = random.randint(3, img_deps-block_noise_size_z-3)
        x[
          noise_x:noise_x+block_noise_size_x, 
          noise_y:noise_y+block_noise_size_y
        #   noise_z:noise_z+block_noise_size_z
          ] = np.random.rand(block_noise_size_x, 
                                                               block_noise_size_y, 
                                                            #    block_noise_size_z, 
                                                               ) * 1.0
        cnt -= 1
    return x

def image_out_painting(x):
    img_rows, img_cols = x.shape
    image_temp = copy.deepcopy(x)
    x = np.random.rand(x.shape[0], x.shape[1], ) * 1.0
    block_noise_size_x = img_rows - random.randint(2*img_rows//7, 4*img_rows//7) # modif 3 4
    block_noise_size_y = img_cols - random.randint(2*img_cols//7, 4*img_cols//7)
    # block_noise_size_z = img_deps - random.randint(3*img_deps//7, 4*img_deps//7)
    noise_x = random.randint(3, img_rows-block_noise_size_x-3)
    noise_y = random.randint(3, img_cols-block_noise_size_y-3)
    # noise_z = random.randint(3, img_deps-block_noise_size_z-3)
    x[
      noise_x:noise_x+block_noise_size_x, 
      noise_y:noise_y+block_noise_size_y
    #   noise_z:noise_z+block_noise_size_z
      ] = image_temp[noise_x:noise_x+block_noise_size_x, 
                                                       noise_y:noise_y+block_noise_size_y
                                                    #    noise_z:noise_z+block_noise_size_z
                                                       ]
    cnt = 4
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)
        block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)
        # block_noise_size_z = img_deps - random.randint(3*img_deps//7, 4*img_deps//7)
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        # noise_z = random.randint(3, img_deps-block_noise_size_z-3)
        x[
          noise_x:noise_x+block_noise_size_x, 
          noise_y:noise_y+block_noise_size_y 
        #   noise_z:noise_z+block_noise_size_z
          ] = image_temp[ noise_x:noise_x+block_noise_size_x, 
                                                           noise_y:noise_y+block_noise_size_y
                                                        #    noise_z:noise_z+block_noise_size_z
                                                           ]
        cnt -= 1
    return x
    
def create_random_patch_mask(batch_size, img_size=256, patch_size=16, mask_ratio=0.75):
    num_patches = (img_size // patch_size) ** 2  # Total number of patches per image
    target_mask_area = int(mask_ratio * img_size * img_size)  # Target area to mask
    
    mask = np.zeros((batch_size, img_size, img_size), dtype=np.uint8) 

    for i in range(batch_size):
        current_mask_area = 0
        patch_indices = np.arange(num_patches)
        np.random.shuffle(patch_indices)  # Shuffle patch indices to select random patches
        
        for idx in patch_indices:
            # Calculate patch's top-left corner
            row = (idx // (img_size // patch_size)) * patch_size
            col = (idx % (img_size // patch_size)) * patch_size
            
            # Add patch to the mask if it doesn't exceed the target area
            if current_mask_area + patch_size * patch_size <= target_mask_area:
                mask[i, row:row + patch_size, col:col + patch_size] = 1
                current_mask_area += patch_size * patch_size

            # Stop if target mask area is reached
            if current_mask_area >= target_mask_area:
                break

    return mask

def generate_pair_mae(img, batch_size, config, status="test", device="cpu") :
    
    img_rows, img_cols = img.shape[1], img.shape[2] #, img.shape[4]

    while True:
        index = [i for i in range(img.shape[0])]
        random.shuffle(index)
        y = img[index[:batch_size]]
        x = copy.deepcopy(y)
        mask = create_random_patch_mask(batch_size, mask_ratio =0.5)
        x = x * (1 - mask[0])
        yield (torch.tensor(x, device=device), torch.tensor(y, device=device))

def generate_pair(img, batch_size, config, status="test", device="cpu") :

    img_rows, img_cols = img.shape[1], img.shape[2] #, img.shape[4]
    while True:
        index = [i for i in range(img.shape[0])]
        random.shuffle(index)
        y = img[index[:batch_size]]
        x = copy.deepcopy(y)
        for n in range(batch_size):
            
            # Autoencoder
            x[n] = copy.deepcopy(y[n])
            
            # Flip
            x[n], y[n] = data_augmentation(x[n], y[n], config.flip_rate)

            # Local Shuffle Pixel
            x[n] = local_pixel_shuffling(x[n], prob=config.local_rate)
            
            # # Apply non-Linear transformation with an assigned probability
            x[n] = nonlinear_transformation(x[n], config.nonlinear_rate)
            
            # Inpainting & Outpainting
            if random.random() < config.paint_rate:
                if random.random() < config.inpaint_rate:
                    # Inpainting
                    x[n] = image_in_painting(x[n])
                else:
                    # Outpainting
                    x[n] = image_out_painting(x[n])

        # # Save sample images module
        # if config.save_samples is not None and status == "train" and random.random() < 0.01:
        #     n_sample = random.choice( [i for i in range(config.batch_size)] )
        #     sample_1 = np.concatenate((x[n_sample,0,:,:,2*img_deps//6], y[n_sample,0,:,:,2*img_deps//6]), axis=1)
        #     sample_2 = np.concatenate((x[n_sample,0,:,:,3*img_deps//6], y[n_sample,0,:,:,3*img_deps//6]), axis=1)
        #     sample_3 = np.concatenate((x[n_sample,0,:,:,4*img_deps//6], y[n_sample,0,:,:,4*img_deps//6]), axis=1)
        #     sample_4 = np.concatenate((x[n_sample,0,:,:,5*img_deps//6], y[n_sample,0,:,:,5*img_deps//6]), axis=1)
        #     final_sample = np.concatenate((sample_1, sample_2, sample_3, sample_4), axis=0)
        #     final_sample = final_sample * 255.0
        #     final_sample = final_sample.astype(np.uint8)
        #     file_name = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(10)])+'.'+config.save_samples
        #     imageio.imwrite(os.path.join(config.sample_path, config.exp_name, file_name), final_sample)

        yield (torch.tensor(x, device=device), torch.tensor(y, device=device))
