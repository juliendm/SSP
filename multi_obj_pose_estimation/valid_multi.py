import os
os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import matplotlib.pyplot as plt
import scipy.misc
import warnings
import sys
import argparse
warnings.filterwarnings("ignore")
from torch.autograd import Variable
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import json

from PIL import Image

import dataset_multi
from darknet_multi import Darknet
from utils_multi import *
from cfg import parse_cfg
from MeshPly import MeshPly

edges_corners = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]

def valid(datacfg, cfgfile, weightfile, visualize=False):
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    # Parse data configuration files
    data_options = read_data_cfg(datacfg)
    valid_images = data_options['valid']
    im_width     = int(data_options['im_width'])
    im_height    = int(data_options['im_height']) 
    fx           = float(data_options['fx'])
    fy           = float(data_options['fy'])
    u0           = float(data_options['u0'])
    v0           = float(data_options['v0'])
    
    # Parse net configuration file
    net_options   = parse_cfg(cfgfile)[0]
    loss_options  = parse_cfg(cfgfile)[-1]
    conf_thresh   = float(net_options['conf_thresh'])
    num_keypoints = int(net_options['num_keypoints'])
    num_classes   = int(loss_options['classes'])
    num_anchors   = int(loss_options['num'])
    anchors       = [float(anchor) for anchor in loss_options['anchors'].split(',')]

    # Read object model information, get 3D bounding box corners, get intrinsics
    #diam                  = float(data_options['diam'])
    intrinsic_calibration = get_camera_intrinsic(u0, v0, fx, fy) # camera params

    # Network I/O params
    num_labels = 2*num_keypoints+3 # +2 for width, height, +1 for object class
    errs_2d = []  # to save
    with open(valid_images) as fp:     # validation file names
        tmp_files = fp.readlines()
        valid_files = [item.rstrip() for item in tmp_files]

    # Compute-related Parameters
    use_cuda = True # whether to use cuda or no
    kwargs = {'num_workers': 4, 'pin_memory': True} # number of workers etc.

    # Specicy model, load pretrained weights, pass to GPU and set the module in evaluation mode
    model = Darknet(cfgfile)
    model.load_weights(weightfile)
    model.cuda()
    model.eval()

    # Get the dataloader for the test dataset
    valid_dataset = dataset_multi.listDataset(valid_images, shape=(model.width, model.height), shuffle=False, transform=transforms.Compose([transforms.ToTensor(),]))
    test_loader   = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, **kwargs) 

    # Iterate through test batches (Batch size for test data is 1)
    logging('Validation ...')
    for batch_idx, (data, target) in enumerate(test_loader):

        t1 = time.time()
        # Pass data to GPU
        if use_cuda:
            data = data.cuda()
            target = target.cuda()
        
        # Wrap tensors in Variable class, set volatile=True for inference mode and to use minimal memory during inference
        data = Variable(data, volatile=True)
        t2 = time.time()
        
        # Forward pass
        output = model(data).data  
        t3 = time.time()
        
        # Using confidence threshold, eliminate low-confidence predictions
        trgt = target[0].view(-1, num_labels)

        print('WARNING: SHOULD BE MORE GENERAL, CASE BY CASE, FOR CORRESPONDING CLASSES: REPLACE 0')
        all_boxes = get_multi_region_boxes(output, conf_thresh, num_classes, num_keypoints, anchors, num_anchors, int(0), only_objectness=0)        
        t4 = time.time()
        
        # Iterate through all images in the batch
        for i in range(output.size(0)):

            if visualize:
                # Visualize
                fig = plt.figure(figsize=(15, 10), dpi=100)
                img = data[i, :, :, :]
                img = img.cpu().numpy().squeeze()
                img = np.transpose(img, (1, 2, 0))
                print(img.shape)

                #img = Image.fromarray(img, 'RGB')
                img = Image.fromarray((img * 255).astype(np.uint8)).resize((im_width,im_height))
                # size = tuple((np.array(im.size) * 0.99999).astype(int))
                # new_image = np.array(im.resize(size, PIL.Image.BICUBIC))


                plt.xlim((0, im_width))
                plt.ylim((0, im_height))
                plt.imshow(img)
            
            # For each image, get all the predictions
            boxes   = all_boxes[i]
            
            # For each image, get all the targets (for multiple object pose estimation, there might be more than 1 target per image)
            truths  = target[i].view(-1, num_labels)
            
            # Get how many object are present in the scene
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
                
                # Denormalize the corner predictions 
                corners2D_gt = np.array(np.reshape(box_gt[:2*num_keypoints], [-1, 2]), dtype='float32')
                corners2D_gt[:, 0] = corners2D_gt[:, 0] * im_width
                corners2D_gt[:, 1] = corners2D_gt[:, 1] * im_height
                corners2D_gt_corrected = corners2D_gt #fix_corner_order(corners2D_gt) # Fix the order of corners
                
                # Compute [R|t] by pnp
                objpoints3D = np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32')
                K = np.array(intrinsic_calibration, dtype='float32')
                R_gt, t_gt = pnp(objpoints3D,  corners2D_gt_corrected, K)
                
                # Compute pixel error
                Rt_gt        = np.concatenate((R_gt, t_gt), axis=1)
                proj_2d_gt   = compute_projection(vertices, Rt_gt, intrinsic_calibration) 
                proj_corners_gt = np.transpose(compute_projection(corners3D, Rt_gt, intrinsic_calibration)) 

                if visualize:
                    # Projections
                    for edge in edges_corners:
                        plt.plot(proj_corners_gt[edge, 0], proj_corners_gt[edge, 1], color='g', linewidth=1.0)

            # Iterate through each ground-truth object
            for k in range(len(boxes)):
                # Read object model information, get 3D bounding box corners
                model_id = int(truths[k][0])
                with open('../../baidu_data/models/json/%s.json' % car_id2name[model_id]) as json_file:
                    mesh = json.load(json_file)
                # Note: already extended with "ones" for translation transformation
                vertices      = np.c_[np.array(mesh['vertices']), np.ones((len(mesh['vertices']), 1))].transpose()
                corners3D     = get_3D_corners(vertices)   

                box_pr        = boxes[k]
                
                if box_pr[2*num_keypoints] > 0.3:

                    # Denormalize the corner predictions 
                    corners2D_pr = np.array(np.reshape(box_pr[:2*num_keypoints], [-1, 2]), dtype='float32')            
                    corners2D_pr[:, 0] = corners2D_pr[:, 0] * im_width
                    corners2D_pr[:, 1] = corners2D_pr[:, 1] * im_height
                    
                    # Compute [R|t] by pnp
                    objpoints3D = np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32')
                    K = np.array(intrinsic_calibration, dtype='float32')
                    R_pr, t_pr = pnp(objpoints3D,  corners2D_pr, K)
                    
                    # Compute pixel error
                    Rt_pr        = np.concatenate((R_pr, t_pr), axis=1)
                    proj_2d_pred = compute_projection(vertices, Rt_pr, intrinsic_calibration) 
                    proj_corners_pr = np.transpose(compute_projection(corners3D, Rt_pr, intrinsic_calibration)) 

                    if visualize:
                        # Projections
                        for edge in np.array(edges_corners):
                            plt.plot(proj_corners_pr[edge, 0], proj_corners_pr[edge, 1], color='b', linewidth=1.0)
                            plt.plot(corners2D_pr[edge+1, 0], corners2D_pr[edge+1, 1], color='r', linewidth=1.0)

            if visualize:
                plt.gca().invert_yaxis()
                plt.show()

        t5 = time.time()

    # Compute 2D projection score
    eps = 1e-5
    for px_threshold in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        acc = len(np.where(np.array(errs_2d) <= px_threshold)[0]) * 100. / (len(errs_2d)+eps)
        # Print test statistics
        logging('   Acc using {} px 2D Projection = {:.2f}%'.format(px_threshold, acc))

if __name__ == '__main__' and __package__ is None:

    parser = argparse.ArgumentParser(description='SingleShotPose')
    parser.add_argument('--modelcfg', type=str, default='cfg/yolo-pose-multi.cfg') # network config
    parser.add_argument('--initweightfile', type=str, default='backup_multi/model_backup.weights') # initialization weights
    args = parser.parse_args()
    datacfg = 'cfg/ape_occlusion.data'
    valid(datacfg, args.modelcfg, args.initweightfile)
    datacfg = 'cfg/can_occlusion.data'
    valid(datacfg, args.modelcfg, args.initweightfile)
    datacfg = 'cfg/cat_occlusion.data'
    valid(datacfg, args.modelcfg, args.initweightfile)
    datacfg = 'cfg/duck_occlusion.data'
    valid(datacfg, args.modelcfg, args.initweightfile)
    datacfg = 'cfg/glue_occlusion.data'
    valid(datacfg, args.modelcfg, args.initweightfile)
    datacfg = 'cfg/holepuncher_occlusion.data'
    valid(datacfg, args.modelcfg, args.initweightfile)

