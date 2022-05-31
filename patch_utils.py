# Adversarial Patch: patch_utils
# utils for patch initialization and mask generation
# Created by Junbo Zhao 2020/3/19

import numpy as np
import torch
from utils import normalizeOutput
from PIL import Image
import pdb

# Initialize the patch
# TODO: Add circle type
def patch_initialization(patch_type='rectangle', image_size=(3, 128, 128), noise_percentage=0.03,act8bitmode = False):
    if patch_type == 'rectangle':
        mask_length = int((noise_percentage * image_size[1] * image_size[2])**0.5)
        if(act8bitmode):
            #for 8bit mode
            patch = np.floor((np.random.rand(image_size[0], mask_length, mask_length) - 0.5)*256).astype(np.int8)
        else:
            patch = np.floor((np.random.rand(image_size[0], mask_length, mask_length) - 0.5)*(2*(127/128)))
    return patch

# Generate the mask and apply the patch
# TODO: Add circle type
def mask_generation(mask_type='rectangle', patch=None, image_size=(3, 128, 128)):
    applied_patch = np.zeros(image_size)
    if mask_type == 'rectangle':
        # patch rotation
        rotation_angle = np.random.choice(4)
        for i in range(patch.shape[0]):
            patch[i] = np.rot90(patch[i], rotation_angle)  # The actual rotation angle is rotation_angle * 90
        # patch location
        x_location, y_location = np.random.randint(low=0, high=image_size[1]-patch.shape[1]), np.random.randint(low=0, high=image_size[2]-patch.shape[2])
        for i in range(patch.shape[0]):
            applied_patch[:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]] = patch
    mask = applied_patch.copy()
    mask[mask != 0] = 1.0
    return applied_patch, mask, x_location, y_location

# Test the patch on dataset
def test_patch(patch_type, target, patch, test_loader, model,args):
    model.eval()
    test_total, test_actual_total, test_success = 0, 0, 0
    if(args.act_mode_8bit):
       datamin = -128
       datamax = 127
    else:
       datamin = -1
       datamax = 127/128
    for (image, label) in test_loader:
        test_total += label.shape[0]
        assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
        assert image.min() >= datamin, 'input should be larger than -128'
        assert image.max() <= datamax, 'input should be less than 128'
        #image = image.to(args.device)
        image = image.to(args.device)
        label = label.to(args.device)
        output = model(image)
        output = normalizeOutput(output,model)
        _, predicted = torch.max(output.data, 1)
        if predicted[0] == label and predicted[0].data.cpu().numpy() != target:
            #if this sample is not the target class we want to spoof then insert our patch into the image and evaluate the model on the image including the patch. if the patch changes the output of the model to the target we want then test succeeds
            test_actual_total += 1
            applied_patch, mask, x_location, y_location = mask_generation(patch_type, patch, image_size=(3, 128, 128))
            applied_patch = torch.from_numpy(applied_patch)
            mask = torch.from_numpy(mask)
            #todo we need to change this to match expected input range for max78000 and normalize input data to correct data range after this
            perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
            assert image.min() >= datamin, 'input should be larger than -128'
            assert image.max() <= datamax, 'input should be less than 128'
            perturbated_image = perturbated_image.to(args.device)
            output = model(perturbated_image)
            output = normalizeOutput(output,model)
            _, predicted = torch.max(output.data, 1)
            if predicted[0].data.cpu().numpy() == target:
                test_success += 1
                # if you want to see the successful test image
                # pdb.set_trace()
                # patchimg=np.moveaxis(perturbated_image[0][:][:][:].numpy().astype('uint8'),0,2)
                # patchimgunsigned = patchimg+128
                # imgfile=Image.fromarray(patchimgunsigned.astype('uint8'),'RGB')
                # imgfile.save("training_pictures/" + str(test_success) + " successfulperturbedimg.png")
                # patchimg=np.moveaxis(image[0][:][:][:].numpy().astype('uint8'),0,2)
                # patchimgunsigned = patchimg+128
                # imgfile=Image.fromarray(patchimgunsigned.astype('uint8'),'RGB')
                # imgfile.save("training_pictures/" + str(test_success) + " groundtruth.png")
                
    return test_success / test_actual_total
    

def save_patch(patchnumpy, patch_file_path, act8bitmode):
    patchimg=np.moveaxis(patchnumpy,0,2)
    if(act8bitmode):
        patchimgunsigned = patchimg+128
    else:
        patchimgunsigned = (patchimg*128)+128
    imgfile=Image.fromarray(patchimgunsigned.astype('uint8'),'RGB')
    imgfile.save(patch_file_path)

#returns patch as numpy array
def load_patch(patch_file_path, act8bitmode):
    #print("Loading patch file " + patch_file_path)
    im = Image.open(patch_file_path)
    rgb_im = im.convert('RGB')
    pilarr = np.array(rgb_im,dtype=np.uint8)
    #print(np.shape(pilarr))
    #we use color channel first
    pilarr=np.moveaxis(pilarr,2,0)
    if(act8bitmode):
        #patch = pilarr - 128
        patch =pilarr.astype(np.float32)
        patch = patch - 128
    else:
        #patch = (pilarr - 128)/128
        patch = pilarr.astype(np.float32)
        patch = (patch -128)/128
    #debug
    #save_patch(patch,patch_file_path+"debug.png",act8bitmode)
    return patch