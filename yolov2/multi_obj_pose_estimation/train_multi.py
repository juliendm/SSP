from __future__ import print_function
import os
os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import random
import math
import shutil
from torchvision import datasets, transforms
from torch.autograd import Variable # Useful info about autograd: http://pytorch.org/docs/master/notes/autograd.html

import matplotlib.pyplot as plt
import json

from darknet_multi import Darknet
from MeshPly import MeshPly
import json
from utils_multi import *    
from cfg import parse_cfg
from region_loss_multi import RegionLoss
import dataset_multi

# Adjust learning rate during training, learning schedule can be changed in network config file
def adjust_learning_rate(optimizer, batch, learning_rate, steps, scales, batch_size):
    lr = learning_rate
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr/batch_size
    return lr


def eval(model, niter, datacfg, modelcfg, testing_iters, testing_accuracies, testing_errors_pixel):
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    # Parse data configuration file
    data_options = read_data_cfg(datacfg)
    valid_images = data_options['valid']
    gpus         = data_options['gpus']  
    num_workers  = int(data_options['num_workers'])
    backupdir    = data_options['backup']
    im_width     = int(data_options['im_width'])
    im_height    = int(data_options['im_height']) 
    fx           = float(data_options['fx'])
    fy           = float(data_options['fy'])
    u0           = float(data_options['u0'])
    v0           = float(data_options['v0'])

    internal_calibration = get_camera_intrinsic(u0, v0, fx, fy)

    # Parse network and training configuration parameters
    net_options   = parse_cfg(modelcfg)[0]
    loss_options  = parse_cfg(modelcfg)[-1]
    batch_size    = int(net_options['batch'])
    max_batches   = int(net_options['max_batches'])
    max_epochs    = int(net_options['max_epochs'])
    learning_rate = float(net_options['learning_rate'])
    momentum      = float(net_options['momentum'])
    decay         = float(net_options['decay'])
    conf_thresh   = float(net_options['conf_thresh'])
    num_keypoints = int(net_options['num_keypoints'])
    num_classes   = int(loss_options['classes'])
    num_anchors   = int(loss_options['num'])
    steps         = [float(step) for step in net_options['steps'].split(',')]
    scales        = [float(scale) for scale in net_options['scales'].split(',')]
    anchors       = [float(anchor) for anchor in loss_options['anchors'].split(',')]



    cur_model = model.module
    #cur_model = model


    # Get validation file names
    with open(valid_images) as fp:
        tmp_files = fp.readlines()
        valid_files = [item.rstrip() for item in tmp_files]
        
    # Specify model, load pretrained weights, pass to GPU and set the module in evaluation mode
    model.eval()
    
    # Get the parser for the test dataset
    valid_dataset = dataset_multi.listDataset(valid_images, shape=(cur_model.width, cur_model.height),
                       shuffle=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ]))
    valid_batchsize = 1

    # Specify the number of workers for multiple processing, get the dataloader for the test dataset
    kwargs = {'num_workers': 4, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_batchsize, shuffle=False, **kwargs) 

    # Parameters
    num_classes          = cur_model.num_classes
    anchors              = cur_model.anchors
    num_anchors          = cur_model.num_anchors
    testing_error_pixel  = 0.0
    testing_samples      = 0.0
    errs_2d              = []

    use_cuda      = True
    num_labels    = num_keypoints*2+3 # + 2 for image width, height, +1 for image class

    logging("   Number of validation samples: %d" % len(test_loader.dataset))
    # Iterate through test examples 
    for batch_idx, (data, target) in enumerate(test_loader):
        t1 = time.time()
        
        # Pass the data to GPU
        if use_cuda:
            data = data.cuda()
            target = target.cuda()
        
        # Wrap tensors in Variable class, set volatile=True for inference mode and to use minimal memory during inference
        data = Variable(data, volatile=True)
        t2 = time.time()
        
        # Formward pass
        output = model(data).data  
        t3 = time.time()
        
        # Using confidence threshold, eliminate low-confidence predictions
        trgt = target[0].view(-1, num_labels)

        print('WARNING: SHOULD BE MORE GENERAL, CASE BY CASE, FOR CORRESPONDING CLASSES')
        all_boxes = get_multi_region_boxes(output, conf_thresh, num_classes, num_keypoints, anchors, num_anchors, int(trgt[0][0]), only_objectness=0)    
        t4 = time.time()

        # Iterate through all batch elements
        for i in range(output.size(0)):

            # For each image, get all the predictions
            boxes   = all_boxes[i]

            # For each image, get all the targets (for multiple object pose estimation, there might be more than 1 target per image)
            truths  = target[i].view(-1, num_labels)

            # Get how many objects are present in the scene
            num_gts = truths_length(truths)

            # Iterate through each ground-truth object
            for k in range(num_gts):

                # Read object model information, get 3D bounding box corners
                model_id = int(truths[k][0])
                with open('../../baidu_data/models/json/%s.json' % car_id2name[model_id]) as json_file:
                    mesh = json.load(json_file)
                # Note: already extended with "ones" for translation transformation
                vertices      = np.c_[np.array(mesh['vertices']), np.ones((len(mesh['vertices']), 1))].transpose()
                corners3D     = get_3D_corners(vertices)   

                box_gt = list()
                for j in range(1, num_labels):
                    box_gt.append(truths[k][j])
                box_gt.extend([1.0, 1.0])
                box_gt.append(truths[k][0])
                
                # If the prediction has the highest confidence, choose it as our prediction
                best_conf_est = -sys.maxsize
                print('WARNING: SHOULD BE MORE GENERAL, CASE BY CASE, FOR ALL BOXES INSTEAD OF BEST ONE')
                for j in range(len(boxes)):
                    if (boxes[j][2*num_keypoints] > best_conf_est) and (boxes[j][2*num_keypoints+2] == int(truths[k][0])):
                        best_conf_est = boxes[j][2*num_keypoints]
                        box_pr        = boxes[j]
                        match         = corner_confidence(box_gt[:2*num_keypoints], torch.FloatTensor(boxes[j][:2*num_keypoints]))

                # Denormalize the corner predictions 
                corners2D_gt = np.array(np.reshape(box_gt[:2*num_keypoints], [num_keypoints, 2]), dtype='float32')
                corners2D_pr = np.array(np.reshape(box_pr[:2*num_keypoints], [num_keypoints, 2]), dtype='float32')
                corners2D_gt[:, 0] = corners2D_gt[:, 0] * im_width
                corners2D_gt[:, 1] = corners2D_gt[:, 1] * im_height               
                corners2D_pr[:, 0] = corners2D_pr[:, 0] * im_width
                corners2D_pr[:, 1] = corners2D_pr[:, 1] * im_height

                print('WARNING: DOES FIX CORNER ORDER STILL APPLY???????????????')
                corners2D_gt_corrected = corners2D_gt # fix_corner_order(corners2D_gt) # Fix the order of the corners in OCCLUSION

                # Compute [R|t] by pnp
                objpoints3D = np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32')
                K = np.array(internal_calibration, dtype='float32')
                R_gt, t_gt = pnp(objpoints3D,  corners2D_gt_corrected, K)
                R_pr, t_pr = pnp(objpoints3D,  corners2D_pr, K)
                
                # Compute pixel error
                Rt_gt           = np.concatenate((R_gt, t_gt), axis=1)
                Rt_pr           = np.concatenate((R_pr, t_pr), axis=1)

                proj_corners_gt = np.transpose(compute_projection(corners3D, Rt_gt, internal_calibration)) 
                proj_corners_pr = np.transpose(compute_projection(corners3D, Rt_pr, internal_calibration))

                proj_2d_gt      = compute_projection(vertices, Rt_gt, internal_calibration) 
                proj_2d_pred    = compute_projection(vertices, Rt_pr, internal_calibration)      
                norm            = np.linalg.norm(proj_2d_gt - proj_2d_pred, axis=0)
                pixel_dist      = np.mean(norm)
                errs_2d.append(pixel_dist)

                # Sum errors
                testing_error_pixel += pixel_dist
                testing_samples     += 1

        t5 = time.time()

    # Compute 2D reprojection score
    eps = 1e-5
    for px_threshold in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        acc = len(np.where(np.array(errs_2d) <= px_threshold)[0]) * 100. / (len(errs_2d)+eps)
        logging('   Acc using {} px 2D Projection = {:.2f}%'.format(px_threshold, acc))

    if True:
        logging('-----------------------------------')
        logging('  tensor to cuda : %f' % (t2 - t1))
        logging('         predict : %f' % (t3 - t2))
        logging('get_region_boxes : %f' % (t4 - t3))
        logging('            eval : %f' % (t5 - t4))
        logging('           total : %f' % (t5 - t1))
        logging('-----------------------------------')

    # Register losses and errors for saving later on
    testing_iters.append(niter)
    testing_errors_pixel.append(testing_error_pixel/(float(testing_samples)+eps))
    testing_accuracies.append(acc)



def train(datacfg, modelcfg, initweightfile, pretrain_num_epochs=0, visualize=False):

    # Parse data configuration file
    data_options = read_data_cfg(datacfg)
    trainlist    = data_options['train']
    gpus         = data_options['gpus']  
    num_workers  = int(data_options['num_workers'])
    backupdir    = data_options['backup']
    im_width     = int(data_options['im_width'])
    im_height    = int(data_options['im_height']) 
    fx           = float(data_options['fx'])
    fy           = float(data_options['fy'])
    u0           = float(data_options['u0'])
    v0           = float(data_options['v0'])

    # Parse network and training configuration parameters
    net_options   = parse_cfg(modelcfg)[0]
    loss_options  = parse_cfg(modelcfg)[-1]
    batch_size    = int(net_options['batch'])
    max_batches   = int(net_options['max_batches'])
    max_epochs    = int(net_options['max_epochs'])
    learning_rate = float(net_options['learning_rate'])
    momentum      = float(net_options['momentum'])
    decay         = float(net_options['decay'])
    conf_thresh   = float(net_options['conf_thresh'])
    num_keypoints = int(net_options['num_keypoints'])
    num_classes   = int(loss_options['classes'])
    num_anchors   = int(loss_options['num'])
    steps         = [float(step) for step in net_options['steps'].split(',')]
    scales        = [float(scale) for scale in net_options['scales'].split(',')]
    anchors       = [float(anchor) for anchor in loss_options['anchors'].split(',')]

    # Further params
    if not os.path.exists(backupdir):
        makedirs(backupdir)
    bg_file_names = ['a','b','c'] #get_all_files('../VOCdevkit/VOC2012/JPEGImages')
    nsamples      = file_lines(trainlist)
    use_cuda      = True
    seed          = int(time.time())
    best_acc      = -sys.maxsize
    num_labels    = num_keypoints*2+3 # + 2 for image width, height, +1 for image class

    # Specify which gpus to use
    torch.manual_seed(seed)
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)

    # Specifiy the model and the loss
    model       = Darknet(modelcfg)
    region_loss = RegionLoss(num_keypoints=num_keypoints, num_classes=num_classes, anchors=anchors, num_anchors=num_anchors, pretrain_num_epochs=pretrain_num_epochs)

    # Model settings

    if initweightfile.split('.')[-1] == 'weights':
        model.load_weights(initweightfile)
    else:
        model.load_weights(initweightfile)
        #model.load_weights_until_last(initweightfile)

    for name,param in model.named_parameters():
#        print(name)
        layer_index = int(name.split('.')[1])
        if layer_index > 23:
            break
        param.requires_grad = False

    # for k, v in model.state_dict().items():
    #     print(k,type(v))


    model.print_network()
    #original_seen     = model.seen
    #model.seen        = 0
    print('Already seen:', model.seen)
    region_loss.iter  = model.iter

    region_loss.seen  = model.seen

    processed_batches = model.seen//batch_size
    init_width        = model.width
    init_height       = model.height
    init_epoch        = model.seen//nsamples 

    # Variables to save
    training_iters       = []
    training_losses      = []
    testing_iters        = []
    testing_errors_pixel = []
    testing_accuracies   = []

    # Specify the number of workers
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}

    # Pass the model to GPU
    if use_cuda:
        # model = model.cuda() 
        # cur_model = model
        model = torch.nn.DataParallel(model, device_ids=[0]).cuda() # Multiple GPU parallelism
        cur_model = model.module


    # Get the optimizer
    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if key.find('.bn') >= 0 or key.find('.bias') >= 0:
            params += [{'params': [value], 'weight_decay': 0.0}]
        else:
            params += [{'params': [value], 'weight_decay': decay*batch_size}]
    optimizer = optim.SGD(model.parameters(), lr=learning_rate/batch_size, momentum=momentum, dampening=0, weight_decay=decay*batch_size)
#    optimizer = optim.Adam(model.parameters(), lr=learning_rate/batch_size)

    evaluate = False
    if evaluate:
        logging('evaluating ...')
        test(0, 0)
    else:
        for epoch in range(init_epoch, max_epochs): 

            # TRAIN

            # Initialize timer
            t0 = time.time()

            # Get the dataloader for training dataset
            train_loader = torch.utils.data.DataLoader(dataset_multi.listDataset(trainlist, shape=(init_width, init_height),
                                                                    shuffle=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(),]), 
                                                                    train=True, 
                                                                    seen=cur_model.seen,
                                                                    batch_size=batch_size,
                                                                    num_workers=num_workers, bg_file_names=bg_file_names),
                                                        batch_size=batch_size, shuffle=False, **kwargs)

            # TRAINING
            lr = adjust_learning_rate(optimizer, processed_batches, learning_rate, steps, scales, batch_size)
            logging('epoch %d, processed %d samples, lr %f' % (epoch, epoch * len(train_loader.dataset), lr))
            # Start training
            model.train()
            t1 = time.time()
            avg_time = torch.zeros(9)
            niter = 0
            # Iterate through batches
            for batch_idx, (data, target) in enumerate(train_loader):

                if visualize:
                    def truths_length(truths):
                        for i in range(50):
                            if truths[i][1] == 0:
                                return i
                    edges_corners = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]

                    for i in range(data.size(0)):

                        img = data[i, :, :, :]
                        img = img.numpy().squeeze()
                        img = np.transpose(img, (1, 2, 0))

                        # Visualize
                        plt.xlim((0, 1696))
                        plt.ylim((0, 608))
                        plt.imshow(img)
                        
                        truths  = target[i].view(-1, num_labels)
                        
                        num_gts = truths_length(truths)
                        for k in range(num_gts): 

                        #     box_gt = list()
                        #     for j in range(1, num_labels):
                        #         box_gt.append(truths[k][j])
                        #     box_gt.extend([1.0, 1.0])
                        #     box_gt.append(truths[k][0])
                            
                        #     # Denormalize the corner predictions 
                        #     corners2D_gt = np.array(np.reshape(box_gt[:2*num_keypoints], [-1, 2]), dtype='float32')
                        #     corners2D_gt[:, 0] = corners2D_gt[:, 0] * im_width
                        #     corners2D_gt[:, 1] = corners2D_gt[:, 1] * im_height               
                        #     corners2D_gt_corrected = fix_corner_order(corners2D_gt) # Fix the order of corners
                            
                        #     # Compute [R|t] by pnp
                        #     intrinsic_calibration = get_camera_intrinsic(u0, v0, fx, fy) 
                        #     objpoints3D = np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32')
                        #     K = np.array(intrinsic_calibration, dtype='float32')
                        #     R_gt, t_gt = pnp(objpoints3D,  corners2D_gt_corrected, K)
                            
                        #     # Compute pixel error
                        #     Rt_gt        = np.concatenate((R_gt, t_gt), axis=1)
                        #     proj_corners_gt = np.transpose(compute_projection(corners3D, Rt_gt, intrinsic_calibration)) 


                            plt.scatter(truths[k, 1]*1696, truths[k, 2]*608, s=20, color='b')
                            corners_gt = truths[k, 3:19].reshape(8,2)
                            for edge in edges_corners:
                                plt.plot(corners_gt[edge, 0]*1696, corners_gt[edge, 1]*608, color='g', linewidth=1)

                        plt.gca().invert_yaxis()
                        plt.show()


                t2 = time.time()
                # adjust learning rate
                adjust_learning_rate(optimizer, processed_batches, learning_rate, steps, scales, batch_size)
                processed_batches = processed_batches + 1
                # Pass the data to GPU
                if use_cuda:
                    data = data.cuda()
                t3 = time.time()
                # Wrap tensors in Variable class for automatic differentiation
                data, target = Variable(data), Variable(target)
                t4 = time.time()
                # Zero the gradients before running the backward pass
                optimizer.zero_grad()
                t5 = time.time()
                # Forward pass
                output = model(data)
                t6 = time.time()

                region_loss.seen = region_loss.seen + data.data.size(0)

                # Compute loss, grow an array of losses for saving later on
                loss = region_loss(output, target, epoch)
                training_iters.append(epoch * math.ceil(len(train_loader.dataset) / float(batch_size) ) + niter)
                training_losses.append(convert2cpu(loss.data))
                niter += 1
                t7 = time.time()
                # Backprop: compute gradient of the loss with respect to model parameters
                loss.backward()
                t8 = time.time()
                # Update weights
                optimizer.step()
                t9 = time.time()
                # Print time statistics
                if False and batch_idx > 1:
                    avg_time[0] = avg_time[0] + (t2-t1)
                    avg_time[1] = avg_time[1] + (t3-t2)
                    avg_time[2] = avg_time[2] + (t4-t3)
                    avg_time[3] = avg_time[3] + (t5-t4)
                    avg_time[4] = avg_time[4] + (t6-t5)
                    avg_time[5] = avg_time[5] + (t7-t6)
                    avg_time[6] = avg_time[6] + (t8-t7)
                    avg_time[7] = avg_time[7] + (t9-t8)
                    avg_time[8] = avg_time[8] + (t9-t1)
                    print('-------------------------------')
                    print('       load data : %f' % (avg_time[0]/(batch_idx)))
                    print('     cpu to cuda : %f' % (avg_time[1]/(batch_idx)))
                    print('cuda to variable : %f' % (avg_time[2]/(batch_idx)))
                    print('       zero_grad : %f' % (avg_time[3]/(batch_idx)))
                    print(' forward feature : %f' % (avg_time[4]/(batch_idx)))
                    print('    forward loss : %f' % (avg_time[5]/(batch_idx)))
                    print('        backward : %f' % (avg_time[6]/(batch_idx)))
                    print('            step : %f' % (avg_time[7]/(batch_idx)))
                    print('           total : %f' % (avg_time[8]/(batch_idx)))
                t1 = time.time()
            t1 = time.time()
            niter = epoch * math.ceil(len(train_loader.dataset) / float(batch_size) ) + niter - 1 


            # SAVE

            if (epoch+1) % 5 == 0:

                logging('save training stats to %s/costs_%d.npz' % (backupdir, (region_loss.seen)))

                np.savez(os.path.join(backupdir, "costs_%d.npz" % (region_loss.seen)),
                    training_iters=training_iters,
                    training_losses=training_losses,
                    testing_iters=testing_iters,
                    testing_accuracies=testing_accuracies,
                    testing_errors_pixel=testing_errors_pixel) 

                # if (np.mean(testing_accuracies[-6:]) > best_acc ): # testing for 6 different objects
                #     best_acc = np.mean(testing_accuracies[-6:]) 
                #     logging('best model so far!')

                logging('save weights to %s/model_%d.weights' % (backupdir, (region_loss.seen)))
                cur_model.seen = region_loss.seen
                cur_model.save_weights('%s/model_%d.weights' % (backupdir, (region_loss.seen)))

            # VALID

            if (epoch+1) % 100 == 0:

                logging("Validation...")
                eval(model, niter, datacfg, modelcfg, testing_iters, testing_accuracies, testing_errors_pixel)

if __name__ == "__main__":

    # Parse command window input
    parser = argparse.ArgumentParser(description='SingleShotPose')
    parser.add_argument('--datacfg', type=str, default='cfg/occlusion.data') # data config
    parser.add_argument('--modelcfg', type=str, default='cfg/yolo-pose-multi.cfg') # network config
    parser.add_argument('--initweightfile', type=str, default='backup_multi/init.weights') # initialization weights
    parser.add_argument('--pretrain_num_epochs', type=int, default=0) # how many epoch to pretrain
    args                = parser.parse_args()
    datacfg             = args.datacfg
    modelcfg            = args.modelcfg
    initweightfile      = args.initweightfile
    pretrain_num_epochs = args.pretrain_num_epochs

    train(datacfg, modelcfg, initweightfile, pretrain_num_epochs)


