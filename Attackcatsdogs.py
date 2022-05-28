# Adversarial Patch Attack
# Created by Junbo Zhao 2020/3/17

"""
Reference:
[1] Tom B. Brown, Dandelion Mané, Aurko Roy, Martín Abadi, Justin Gilmer
    Adversarial Patch. arXiv:1712.09665
"""
import pdb
from PIL import Image
import copy
import fnmatch
import logging
import operator
import os
import sys
import time
import traceback
from collections import OrderedDict
from functools import partial
from pydoc import locate

import numpy as np

import matplotlib
from pkg_resources import parse_version

# TensorFlow 2.x compatibility
try:
    import tensorboard  # pylint: disable=import-error
    import tensorflow  # pylint: disable=import-error
    tensorflow.io.gfile = tensorboard.compat.tensorflow_stub.io.gfile
except (ModuleNotFoundError, AttributeError):
    pass

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch import nn
from torch.backends import cudnn

# pylint: disable=wrong-import-order
import distiller
import examples.auto_compression.amc as adc
import shap
import torchnet.meter as tnt
from distiller import apputils, model_summaries
from distiller.data_loggers import PythonLogger, TensorBoardLogger
# pylint: disable=no-name-in-module
from distiller.data_loggers.collector import (QuantCalibrationStatsCollector,
                                              RecordsActivationStatsCollector,
                                              SummaryActivationStatsCollector, collectors_context)
from distiller.quantization.range_linear import PostTrainLinearQuantizer

# pylint: enable=no-name-in-module
import ai8x
import ai8x_nas
import datasets
import nnplot
import parse_qat_yaml
import parsecmd
import sample
from nas import parse_nas_yaml



#target label is the adversarial label we want ("robots" class)

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import models

import argparse
import csv
import os
import numpy as np

from patch_utils import*
from utils import*


use_load_patch = True
evaluate_only = True
patch_file_path = "training_pictures/loaded_patch.png"

imgheight = 128
imgwidth = 128

#hardcoding the attack argparser to use the argparser for max78000 instead
# parser = argparse.ArgumentParser()
# parser.add_argument('--batch_size', type=int, default=1, help="batch size")
# parser.add_argument('--num_workers', type=int, default=2, help="num_workers")
# parser.add_argument('--train_size', type=int, default=2000, help="number of training images")
# parser.add_argument('--test_size', type=int, default=2000, help="number of test images")
# parser.add_argument('--noise_percentage', type=float, default=0.1, help="percentage of the patch size compared with the image size")
# parser.add_argument('--probability_threshold', type=float, default=0.9, help="minimum target probability")
# parser.add_argument('--lr', type=float, default=1.0, help="learning rate")
# parser.add_argument('--max_iteration', type=int, default=1000, help="max iteration")
# parser.add_argument('--target', type=int, default=859, help="target label")
# parser.add_argument('--epochs', type=int, default=20, help="total epoch")
# parser.add_argument('--data_dir', type=str, default='/datasets/imgNet/imagenet1k_valid_dataset/', help="dir of the dataset")
# parser.add_argument('--patch_type', type=str, default='rectangle', help="type of the patch")
# parser.add_argument('--GPU', type=str, default='0', help="index pf used GPU")
# parser.add_argument('--log_dir', type=str, default='patch_attack_log.csv', help='dir of the log')
# args = parser.parse_args()

batch_size = 1
num_workers = 2
train_size = 2000
test_size = 2000
noise_percentage = 0.05
probability_threshold = 0.99
#lr = 1.0
lr = .01
max_iteration = 1000
#max_iteration = 100000
target = 1   #what the patch should look like: 0 cat 1 dog
#epochs = 20
epochs = 100
data_dir = 'pathtodataset'
patch_type = 'rectangle'
GPU = '0'
log_dir = 'patch_attack_log.csv'

###########################################################################
# from range_linear_ai84 import PostTrainLinearQuantizerAI84

#script_dir = os.path.dirname('/c/MaximSDK/Tools/ai8x-tools/ai8x-training')
def main():
    """main"""
    script_dir = os.path.dirname(__file__)
    global msglogger  # pylint: disable=global-statement
    matplotlib.use("pgf")

    # Logger handle
    msglogger = None

    # Globals
    weight_min = None
    weight_max = None
    weight_count = None
    weight_sum = None
    weight_stddev = None
    weight_mean = None

    #global msglogger  # pylint: disable=global-statement

    supported_models = []
    supported_sources = []
    model_names = []
    dataset_names = []



    # Dynamically load models
    for _, _, files in sorted(os.walk('models')):
        for name in sorted(files):
            if fnmatch.fnmatch(name, '*.py'):
                fn = 'models.' + name[:-3]
                m = locate(fn)
                try:
                    for i in m.models:
                        i['module'] = fn
                    supported_models += m.models
                    model_names += [item['name'] for item in m.models]
                except AttributeError:
                    # Skip files that don't have 'models' or 'models.name'
                    pass

    # Dynamically load datasets
    for _, _, files in sorted(os.walk('datasets')):
        for name in sorted(files):
            if fnmatch.fnmatch(name, '*.py'):
                ds = locate('datasets.' + name[:-3])
                try:
                    supported_sources += ds.datasets
                    dataset_names += [item['name'] for item in ds.datasets]
                except AttributeError:
                    # Skip files that don't have 'datasets' or 'datasets.name'
                    pass
                    


    # Parse arguments
    args = parsecmd.get_parser(model_names, dataset_names).parse_args()


    # Set hardware device
    ai8x.set_device(args.device, args.act_mode_8bit, args.avg_pool_rounding)

    if args.epochs is None:
        args.epochs = 90

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    if args.optimizer is None:
        if not args.evaluate:
            print('WARNING: --optimizer not set, selecting SGD.')
        args.optimizer = 'SGD'

    if args.lr is None:
        if not args.evaluate:
            print('WARNING: Initial learning rate (--lr) not set, selecting 0.1.')
        args.lr = 0.1

    msglogger = apputils.config_pylogger(os.path.join(script_dir, 'logging.conf'), args.name,
                                         args.output_dir)

    # Log various details about the execution environment.  It is sometimes useful
    # to refer to past experiment executions and this information may be useful.
    apputils.log_execution_env_state(args.compress, msglogger.logdir)
    msglogger.debug("Distiller: %s", distiller.__version__)

    start_epoch = 0
    ending_epoch = args.epochs
    perf_scores_history = []

    if args.evaluate and args.shap == 0:
        args.deterministic = True
    if args.deterministic:
        # torch.set_deterministic(True)
        distiller.set_deterministic(args.seed)  # For experiment reproducability
        if args.seed is not None:
            distiller.set_seed(args.seed)
    else:
        # Turn on CUDNN benchmark mode for best performance. This is usually "safe" for image
        # classification models, as the input sizes don't change during the run
        # See here:
        # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3
        cudnn.benchmark = True

    if args.cpu or not torch.cuda.is_available():
        if not args.cpu:
            # Print warning if no hardware acceleration
            print("WARNING: CUDA hardware acceleration is not available, training will be slow")
        # Set GPU index to -1 if using CPU
        args.device = 'cpu'
        args.gpus = -1
    else:
        args.device = 'cuda'
        if args.gpus is not None:
            try:
                args.gpus = [int(s) for s in args.gpus.split(',')]
            except ValueError as exc:
                raise ValueError('ERROR: Argument --gpus must be a comma-separated '
                                 'list of integers only') from exc
            available_gpus = torch.cuda.device_count()
            for dev_id in args.gpus:
                if dev_id >= available_gpus:
                    raise ValueError(f'ERROR: GPU device ID {dev_id} requested, but only '
                                     f'{available_gpus} devices available')
            # Set default device in case the first one on the list != 0
            torch.cuda.set_device(args.gpus[0])

    if args.earlyexit_thresholds:
        args.num_exits = len(args.earlyexit_thresholds) + 1
        args.loss_exits = [0] * args.num_exits
        args.losses_exits = []
        args.exiterrors = []

    selected_source = next((item for item in supported_sources if item['name'] == args.dataset))
    args.labels = selected_source['output']
    args.num_classes = len(args.labels)
    if args.num_classes == 1 \
       or ('regression' in selected_source and selected_source['regression']):
        args.regression = True
    dimensions = selected_source['input']
    if len(dimensions) == 2:
        dimensions += (1, )
    args.dimensions = dimensions

    args.datasets_fn = selected_source['loader']
    args.visualize_fn = selected_source['visualize'] \
        if 'visualize' in selected_source else datasets.visualize_data

    if args.regression and args.display_confusion:
        raise ValueError('ERROR: Argument --confusion cannot be used with regression')
    if args.regression and args.display_prcurves:
        raise ValueError('ERROR: Argument --pr-curves cannot be used with regression')
    if args.regression and args.display_embedding:
        raise ValueError('ERROR: Argument --embedding cannot be used with regression')

    model = create_model(supported_models, dimensions, args)

    # if args.add_logsoftmax:
    #     model = nn.Sequential(model, nn.LogSoftmax(dim=1))
    # if args.add_softmax:
    #     model = nn.Sequential(model, nn.Softmax(dim=1))

    compression_scheduler = None
    # Create a couple of logging backends.  TensorBoardLogger writes log files in a format
    # that can be read by Google's Tensor Board.  PythonLogger writes to the Python logger.
    pylogger = PythonLogger(msglogger, log_1d=True)
    all_loggers = [pylogger]
    if args.tblog:
        tflogger = TensorBoardLogger(msglogger.logdir, log_1d=True, comment='_'+args.dataset)

        tflogger.tblogger.writer.add_text('Command line', str(args))

        if dimensions[2] > 1:
            dummy_input = torch.randn((1, ) + dimensions)
        else:  # 1D input
            dummy_input = torch.randn((1, ) + dimensions[:-1])
        tflogger.tblogger.writer.add_graph(model.to('cpu'), (dummy_input, ), False)

        all_loggers.append(tflogger)
        all_tbloggers = [tflogger]
    else:
        tflogger = None
        all_tbloggers = []

    # Capture thresholds for early-exit training
    if args.earlyexit_thresholds:
        msglogger.info('=> using early-exit threshold values of %s', args.earlyexit_thresholds)

    # Get policy for quantization aware training
    qat_policy = parse_qat_yaml.parse(args.qat_policy) \
        if args.qat_policy.lower() != "none" else None

    # Get policy for once for all training policy
    nas_policy = parse_nas_yaml.parse(args.nas_policy) \
        if args.nas and args.nas_policy.lower() != '' else None

    # We can optionally resume from a checkpoint
    optimizer = None
    if args.resumed_checkpoint_path:
        update_old_model_params(args.resumed_checkpoint_path, model)
        if qat_policy is not None:
            checkpoint = torch.load(args.resumed_checkpoint_path,
                                    map_location=lambda storage, loc: storage)
            # pylint: disable=unsubscriptable-object
            if checkpoint.get('epoch', None) >= qat_policy['start_epoch']:
                ai8x.fuse_bn_layers(model)
            # pylint: enable=unsubscriptable-object
        model, compression_scheduler, optimizer, start_epoch = apputils.load_checkpoint(
            model, args.resumed_checkpoint_path, model_device=args.device)
        ai8x.update_model(model)
    elif args.load_model_path:
        update_old_model_params(args.load_model_path, model)
        if qat_policy is not None:
            checkpoint = torch.load(args.load_model_path,
                                    map_location=lambda storage, loc: storage)
            # pylint: disable=unsubscriptable-object
            if checkpoint.get('epoch', None) >= qat_policy['start_epoch']:
                ai8x.fuse_bn_layers(model)
            # pylint: enable=unsubscriptable-object
        model = apputils.load_lean_checkpoint(model, args.load_model_path,
                                              model_device=args.device)
        ai8x.update_model(model)

    if not args.load_serialized and args.gpus != -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpus).to(args.device)

    if args.reset_optimizer:
        start_epoch = 0
        if optimizer is not None:
            optimizer = None
            msglogger.info('\nreset_optimizer flag set: Overriding resumed optimizer and '
                           'resetting epoch count to 0')

    # Define loss function (criterion)
    if not args.regression:
        if 'weight' in selected_source:
            criterion = nn.CrossEntropyLoss(
                torch.Tensor(selected_source['weight'])
            ).to(args.device)
        else:
            criterion = nn.CrossEntropyLoss().to(args.device)
    else:
        criterion = nn.MSELoss().to(args.device)

    if optimizer is None:
        optimizer = create_optimizer(model, args)
        msglogger.info('Optimizer Type: %s', type(optimizer))
        msglogger.info('Optimizer Args: %s', optimizer.defaults)



    #activations_collectors = create_activation_stats_collectors(model, *args.activation_stats)

    if args.qe_calibration:
        msglogger.info('Quantization calibration stats collection enabled:')
        msglogger.info('\tStats will be collected for %.1f%% of test dataset', args.qe_calibration)
        msglogger.info('\tSetting constant seeds and converting model to serialized execution')
        distiller.set_deterministic()
        model = distiller.make_non_parallel_copy(model)
        activations_collectors.update(create_quantization_stats_collector(model))
        args.evaluate = True
        args.effective_test_size = args.qe_calibration
        



    #######


    os.environ["CUDA_VISIBLE_DEVICES"] = GPU

    # Load the model
    #model = models.resnet50(pretrained=True).cuda()
    #model = models.resnet50(pretrained=True)
    print("device "+ args.device)
    model.to(args.device)
    model.eval()

    # Load the datasets
    #train_loader, test_loader = dataloader(args.train_size, args.test_size, args.data_dir, args.batch_size, args.num_workers, 50000)

    # Load the datasets
    #make batch size 1 so we only work on one image at a time
    #train_loader, val_loader, test_loader, _ = apputils.get_data_loaders(
    #        args.datasets_fn, (os.path.expanduser(args.data), args), args.batch_size,
    #        args.workers, args.validation_split, args.deterministic,
    #        args.effective_train_size, args.effective_valid_size, args.effective_test_size)
    batchsz  = 1

    # train_loader, val_loader, test_loader, _ = apputils.get_data_loaders(
            # args.datasets_fn, (os.path.expanduser(args.data), args), batchsz,
            # args.workers, args.validation_split, args.deterministic,
            # args.effective_train_size, args.effective_valid_size, args.effective_test_size)
            
    #args.effective_train_size = 0.03
    #args.effective_valid_size = 0.03
    #args.effective_test_size = 0.03
    
    #normal settings use
    # args.effective_train_size = 0.1
    # args.effective_valid_size = 0.5
    # args.effective_test_size = 0.5
    
    
    #test settings for quick debug
    # args.effective_train_size = 0.003
    # args.effective_valid_size = 0.05
    # args.effective_test_size = 0.05
    
    
    train_loader, val_loader, test_loader, _ = apputils.get_data_loaders(
            args.datasets_fn, (os.path.expanduser(args.data), args), batchsz,
            args.workers, args.validation_split, args.deterministic,
            args.effective_train_size, args.effective_valid_size, args.effective_test_size)
            
    print("training size "+str(len(train_loader)))
    print("validation size "+str(len(val_loader)))
    print("test size "+str(len(test_loader)))
            
    if(args.act_mode_8bit):
        print("Running in 8 bit mode. Evaluation only, training disabled")
        use_load_patch = True
        evaluate_only = True

    # Test the accuracy of model on trainset and testset
    print("running model test")
    #trainset_acc, test_acc = test(model, train_loader,args), test(model, test_loader,args)
    #print('Accuracy of the model on clean trainset and testset is {:.3f}% and {:.3f}%'.format(100*trainset_acc, 100*test_acc))

    # Initialize the patch
    if(use_load_patch):
        print("Loading patch from "+patch_file_path)
        patch = load_patch(patch_file_path, args.act_mode_8bit)
    else:
        patch = patch_initialization(patch_type, image_size=(3, imgheight, imgwidth), noise_percentage=noise_percentage)
    #pdb.set_trace()
    print('The shape of the patch is', patch.shape)
    
    if(evaluate_only):
        train_success_rate = test_patch(patch_type, target, patch, train_loader, model,args)
        print("Patch attack success rate on trainset: {:.3f}%".format(100 * train_success_rate))
        test_success_rate = test_patch(patch_type, target, patch, test_loader, model,args)
        print("Patch attack success rate on testset: {:.3f}%".format(100 * test_success_rate))
        return

    with open(log_dir, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_success", "test_success"])

    best_patch_epoch, best_patch_success_rate = 0, 0

    numinputimages = 0
    
    if(args.act_mode_8bit):
       datamin = -128
       datamax = 127
    else:
       datamin = -1
       datamax = 127/128

    # Generate the patch
    #train
    for epoch in range(epochs):
        train_total, train_actual_total, train_success = 0, 0, 0
        for (image, label) in train_loader:
            train_total += label.shape[0]
            assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
            image = image.to(args.device)
            label = label.to(args.device)
            
            output = model(image)
            output = normalizeOutput(output,model)
            _, predicted = torch.max(output.data, 1)
            if predicted[0] == label and predicted[0].data.cpu().numpy() != target:
                 #here if it is not in our target class then add patch into image and input image with patch added into the model.
                 train_actual_total += 1
                 #create a patch
                 applied_patch, mask, x_location, y_location = mask_generation(patch_type, patch, image_size=(3, imgwidth, imgheight))
                 #applied_patch is numpy array
                 #optimize the patch for this image
                 perturbated_image, applied_patch = patch_attack(args,image, applied_patch, mask, target, probability_threshold, model, lr, max_iteration)
                 perturbated_image = torch.from_numpy(perturbated_image).to(args.device)
                 assert perturbated_image.min() >= datamin, 'input should be larger than -128'
                 assert perturbated_image.max() <=  datamax, 'input should be less than 128'
                 output = model(perturbated_image)
                 output = normalizeOutput(output,model)
                 _, predicted = torch.max(output.data, 1)
                 if predicted[0].data.cpu().numpy() == target:
                     train_success += 1
                 patch = applied_patch[0][:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]]
            numinputimages += 1
            if((numinputimages %10000) == 0):
                print("number of training images used " + str(numinputimages))
        #mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] #transform used in original dataloader
        #plt.imshow(np.clip(np.transpose(patch, (1, 2, 0)) * std + mean, 0, 1))
        assert np.min(patch) >= datamin, 'input should be larger than -128'
        assert np.max(patch) <=  datamax, 'input should be less than 128'
        patchimg=np.moveaxis(patch,0,2)
        if(args.act_mode_8bit):
            patchimgunsigned = patchimg+128
        else:
            patchimgunsigned = (patchimg*128)+128
        imgfile=Image.fromarray(patchimgunsigned.astype('uint8'),'RGB')
        imgfile.save("training_pictures/" + str(epoch) + " patch.png")
        #plt.savefig("training_pictures/" + str(epoch) + " patch.png")
        print("Epoch:{} Patch attack success rate on trainset: {:.3f}%".format(epoch, 100 * train_success / train_actual_total))
        train_success_rate = test_patch(patch_type, target, patch, test_loader, model,args)
        print("Epoch:{} Patch attack success rate on trainset: {:.3f}%".format(epoch, 100 * train_success_rate))
        test_success_rate = test_patch(patch_type, target, patch, test_loader, model,args)
        print("Epoch:{} Patch attack success rate on testset: {:.3f}%".format(epoch, 100 * test_success_rate))

        # Record the statistics
        with open(log_dir, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_success_rate, test_success_rate])

        if test_success_rate > best_patch_success_rate:
            best_patch_success_rate = test_success_rate
            best_patch_epoch = epoch
            #assert np.min(patch) >= -128, 'input should be larger than -128'
            #assert np.max(patch) <=  127, 'input should be less than 128'
            patchimg=np.moveaxis(patch,0,2)
            if(args.act_mode_8bit):
                patchimgunsigned = patchimg+128
            else:
                patchimgunsigned = (patchimg*128)+128
            imgfile=Image.fromarray(patchimgunsigned.astype('uint8'),'RGB')
            imgfile.save("training_pictures/" + str(epoch) + " best_patch.png")
            #plt.imshow(np.clip(np.transpose(patch, (1, 2, 0)) * std + mean, 0, 1))
            #plt.savefig("training_pictures/best_patch.png")
                
        numinputimages = 0 
        # Load the statistics and generate the line
        #log_generation(log_dir)

    print("The best patch is found at epoch {} with success rate {}% on testset".format(best_patch_epoch, 100 * best_patch_success_rate))


# Patch attack via optimization
# According to reference [1], one image is attacked each time
# Assert: applied patch should be a numpy
# Return the final perturbated picture and the applied patch. Their types are both numpy
def patch_attack(args,image, applied_patch, mask, target, probability_threshold, model, lr=1, max_iteration=100):
    if(args.act_mode_8bit):
       datamin = -128
       datamax = 127
    else:
       datamin = -1
       datamax = 127/128
    model.eval()
    applied_patch = torch.from_numpy(applied_patch)
    mask = torch.from_numpy(mask)
    target_probability, count = 0, 0
    perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
    while target_probability < probability_threshold and count < max_iteration:
        count += 1
        # Optimize the patch
        perturbated_image = Variable(perturbated_image.data, requires_grad=True)
        per_image = perturbated_image
        # if(per_image.min() < -128):
           # pdb.set_trace()
        # if(per_image.max() >127):
           # pdb.set_trace()
        assert per_image.min() >= datamin, 'input should be larger than -128'
        assert per_image.max() <=datamax, 'input should be less than 128'
        per_image = per_image.to(args.device)
        #pdb.set_trace()
        output = model(per_image)
        output = normalizeOutput(output,model)
        target_log_softmax = torch.nn.functional.log_softmax(output, dim=1)[0][target]
        target_log_softmax.backward()
        patch_grad = perturbated_image.grad.clone().cpu()
        perturbated_image.grad.data.zero_()
        #pdb.set_trace()
        #follow gradient
        applied_patch = lr * patch_grad + applied_patch.type(torch.FloatTensor)
        applied_patch = torch.clamp(applied_patch, min=datamin, max=datamax)
        # Test the patch
        perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1-mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
        #todo perturbed image input needs to be normalized to data range
        perturbated_image = torch.clamp(perturbated_image, min=datamin, max=datamax)
        perturbated_image = perturbated_image.to(args.device)
        assert perturbated_image.min() >= datamin, 'input should be larger than -128'
        assert perturbated_image.max() <=datamax, 'input should be less than 128'
        output = model(perturbated_image)
        output = normalizeOutput(output,model)
        target_probability = torch.nn.functional.softmax(output, dim=1).data[0][target]
        if(not patch_grad.ne(0).any().item()):
           #gradient is 0
           #print("gradient 0 skipping " + str(count))
           break
    perturbated_image = perturbated_image.cpu().numpy()
    applied_patch = applied_patch.cpu().numpy()
    return perturbated_image, applied_patch



def create_model(supported_models, dimensions, args):
    """Create the model"""
    module = next(item for item in supported_models if item['name'] == args.cnn)

    # Override distiller's input shape detection. This is not a very clean way to do it since
    # we're replacing a protected member.
    distiller.utils._validate_input_shape = (  # pylint: disable=protected-access
        lambda _a, _b: (1, ) + dimensions[:module['dim'] + 1]
    )

    Model = locate(module['module'] + '.' + args.cnn)
    if not Model:
        raise RuntimeError("Model " + args.cnn + " not found\n")

    # Set model paramaters
    if args.act_mode_8bit:
        weight_bits = 8
        bias_bits = 8
        quantize_activation = True
    else:
        weight_bits = None
        bias_bits = None
        quantize_activation = False

    if module['dim'] > 1 and module['min_input'] > dimensions[2]:
        model = Model(pretrained=False, num_classes=args.num_classes,
                      num_channels=dimensions[0],
                      dimensions=(dimensions[1], dimensions[2]),
                      padding=(module['min_input'] - dimensions[2] + 1) // 2,
                      bias=args.use_bias,
                      weight_bits=weight_bits,
                      bias_bits=bias_bits,
                      quantize_activation=quantize_activation).to(args.device)
    else:
        model = Model(pretrained=False, num_classes=args.num_classes,
                      num_channels=dimensions[0],
                      dimensions=(dimensions[1], dimensions[2]),
                      bias=args.use_bias,
                      weight_bits=weight_bits,
                      bias_bits=bias_bits,
                      quantize_activation=quantize_activation).to(args.device)

    return model
    
def update_old_model_params(model_path, model_new):
    """Adds missing model parameters added with default values.
    This is mainly due to the saved checkpoint is from previous versions of the repo.
    New model is saved to `model_path` and the old model copied into the same file_path with
    `__obsolete__` prefix."""
    is_model_old = False
    model_old = torch.load(model_path,
                           map_location=lambda storage, loc: storage)
    # Fix up any instances of DataParallel
    old_dict = model_old['state_dict'].copy()
    for _, k in enumerate(old_dict):
        if k.startswith('module.'):
            model_old['state_dict'][k[7:]] = old_dict[k]
    for new_key, new_val in model_new.state_dict().items():
        if new_key not in model_old['state_dict'] and 'bn' not in new_key:
            is_model_old = True
            model_old['state_dict'][new_key] = new_val
            if 'compression_sched' in model_old:
                if 'masks_dict' in model_old['compression_sched']:
                    model_old['compression_sched']['masks_dict'][new_key] = None

    if is_model_old:
        dir_path, file_name = os.path.split(model_path)
        new_file_name = '__obsolete__' + file_name
        old_model_path = os.path.join(dir_path, new_file_name)
        os.rename(model_path, old_model_path)
        torch.save(model_old, model_path)
        msglogger.info('Model `%s` is old. Missing parameters added with default values!',
                       model_path)
                       
def create_optimizer(model, args):
    """Create the optimizer"""
    if args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        msglogger.info('Unknown optimizer type: %s. SGD is set as optimizer!!!', args.optimizer)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    return optimizer


if __name__ == '__main__':
    try:
        np.set_printoptions(threshold=sys.maxsize, linewidth=190)
        torch.set_printoptions(threshold=sys.maxsize, linewidth=190)

        # import builtins, sys
        # print(distiller.config.__name__)
        # print(distiller.config.__builtins__)
        # # print(distiller.config.__import__)
        # builtins.QuantAwareTrainRangeLinearQuantizerAI84 = \
        #   range_linear_ai84.QuantAwareTrainRangeLinearQuantizerAI84
        # globals()['range_linear_ai84'] = __import__('range_linear_ai84')
        # sys.path.append('/home/robertmuchsel/Documents/Source/ai84')
        # site.addsitedir("/home/robertmuchsel/Documents/Source/ai84")
        main()
    except KeyboardInterrupt:
        print("\n-- KeyboardInterrupt --")
    except Exception:
        raise

