import sys
import os
import time
import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
from scipy import spatial

import struct 
import imghdr 

car_name2id = {'019-SUV': 46, '036-CAR01': 47, '037-CAR02': 16, 'MG-GT-2015': 30, 'Skoda_Fabia-2011': 67, 'aodi-Q7-SUV': 48, 'aodi-a6': 17, 'baojun-310-2017': 0, 'baojun-510': 49, 'baoma-330': 18, 'baoma-530': 19, 'baoma-X5': 50, 'baoshijie-kayan': 51, 'baoshijie-paoche': 20, 'beiqi-huansu-H3': 52, 'benchi-GLK-300': 53, 'benchi-ML500': 54, 'benchi-SUR': 71, 'bentian-fengfan': 21, 'biaozhi-3008': 1, 'biaozhi-408': 22, 'biaozhi-508': 23, 'biaozhi-liangxiang': 2, 'bieke': 37, 'bieke-kaiyue': 24, 'bieke-yinglang-XT': 3, 'biyadi-2x-F0': 4, 'biyadi-F3': 38, 'biyadi-qin': 39, 'biyadi-tang': 72, 'changan-CS35-2012': 73, 'changan-cs5': 74, 'changanbenben': 5, 'changcheng-H6-2016': 75, 'dazhong': 40, 'dazhong-SUV': 76, 'dazhongmaiteng': 41, 'dihao-EV': 42, 'dongfeng-DS5': 6, 'dongfeng-fengguang-S560': 77, 'dongfeng-fengxing-SX6': 78, 'dongfeng-xuetielong-C6': 43, 'dongfeng-yulong-naruijie': 45, 'dongnan-V3-lingyue-2011': 44, 'feiyate': 7, 'fengtian-MPV': 9, 'fengtian-SUV-gai': 56, 'fengtian-liangxiang': 8, 'fengtian-puladuo-06': 55, 'fengtian-weichi-2006': 15, 'fute': 25, 'guangqi-chuanqi-GS4-2015': 57, 'haima-3': 26, 'jianghuai-ruifeng-S3': 58, 'jili-boyue': 59, 'jilixiongmao-2015': 10, 'jipu-3': 60, 'kaidilake-CTS': 27, 'leikesasi': 28, 'lingmu-SX4-2012': 13, 'lingmu-aotuo-2009': 11, 'lingmu-swift': 12, 'linken-SUV': 61, 'lufeng-X8': 62, 'mazida-6-2015': 29, 'oubao': 31, 'qirui-ruihu': 63, 'qiya': 32, 'rongwei-750': 33, 'rongwei-RX5': 64, 'sanling-oulande': 65, 'sikeda-SUV': 66, 'sikeda-jingrui': 14, 'supai-2016': 34, 'xiandai-i25-2016': 68, 'xiandai-suonata': 35, 'yingfeinidi-SUV': 70, 'yingfeinidi-qx80': 69, 'yiqi-benteng-b50': 36}
car_id2name = {v: k for k, v in car_name2id.items()}
car_id2class = {2:0, 6:1, 7:2, 8:3, 9:4, 12:5, 14:6, 16:7, 18:8, 19:9, 20:10, 23:11, 25:12, 27:13, 28:14, 31:15, 32:16, 35:17, 37:18, 40:19, 43:20, 46:21, 47:22, 48:23, 50:24, 51:25, 54:26, 56:27, 60:28, 61:29, 66:30, 70:31, 71:32, 76:33}
car_class2id = {v: k for k, v in car_id2class.items()}

# Create new directory
def makedirs(path):
    if not os.path.exists( path ):
        os.makedirs( path )

# Get all the files within a directory
def get_all_files(directory):
    files = []
    for f in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, f)):
            files.append(os.path.join(directory, f))
        else:
            files.extend(get_all_files(os.path.join(directory, f)))
    return files

# Calculate angular distance between two rotations
def calcAngularDistance(gt_rot, pr_rot):
    rotDiff = np.dot(gt_rot, np.transpose(pr_rot))
    trace = np.trace(rotDiff) 
    return np.rad2deg(np.arccos((trace-1.0)/2.0))

# Get camera intrinsic matrix
def get_camera_intrinsic(u0, v0, fx, fy):
    '''fx, fy: focal length parameters, u0, v0: principal point offset parameters'''
    return np.array([[fx, 0.0, u0], [0.0, fy, v0], [0.0, 0.0, 1.0]])

# Compute the projection of an array of 3D points onto a 2D image given the intrinsics and extrinsics
def compute_projection(points_3D, transformation, internal_calibration):
    projections_2d = np.zeros((2, points_3D.shape[1]), dtype='float32')
    camera_projection = (internal_calibration.dot(transformation)).dot(points_3D)
    projections_2d[0, :] = camera_projection[0, :]/camera_projection[2, :]
    projections_2d[1, :] = camera_projection[1, :]/camera_projection[2, :]
    return projections_2d

# Transform an array of 3D points in the 3D space using extrinsics
def compute_transformation(points_3D, transformation):
    return transformation.dot(points_3D)

# Calculate the diameter of an object model, diameter is defined as the longest distance between all the pairwise distances in the object model
def calc_pts_diameter(pts):
    diameter = -1
    for pt_id in range(pts.shape[0]):
        pt_dup = np.tile(np.array([pts[pt_id, :]]), [pts.shape[0] - pt_id, 1])
        pts_diff = pt_dup - pts[pt_id:, :]
        max_dist = math.sqrt((pts_diff * pts_diff).sum(axis=1).max())
        if max_dist > diameter:
            diameter = max_dist
    return diameter

# Compute adi metric, see https://github.com/thodan/obj_pose_eval/blob/master/obj_pose_eval/pose_error.py for further info
def adi(pts_est, pts_gt):
    nn_index = spatial.cKDTree(pts_est)
    nn_dists, _ = nn_index.query(pts_gt, k=1)
    e = nn_dists.mean()
    return e

# Get the 3D corners of the bounding box surrounding the object model
def get_3D_corners(vertices):
    
    min_x = np.min(vertices[0,:])
    max_x = np.max(vertices[0,:])
    min_y = np.min(vertices[1,:])
    max_y = np.max(vertices[1,:])
    min_z = np.min(vertices[2,:])
    max_z = np.max(vertices[2,:])

    corners = np.array([[min_x, min_y, min_z],
                        [min_x, min_y, max_z],
                        [min_x, max_y, min_z],
                        [min_x, max_y, max_z],
                        [max_x, min_y, min_z],
                        [max_x, min_y, max_z],
                        [max_x, max_y, min_z],
                        [max_x, max_y, max_z]])

    corners = np.concatenate((np.transpose(corners), np.ones((1,8)) ), axis=0)
    return corners

# Compute pose using PnP
def pnp(points_3D, points_2D, cameraMatrix):
    try:
        distCoeffs = pnp.distCoeffs
    except:
        distCoeffs = np.zeros((8, 1), dtype='float32') # 8 distortion-coefficient model

    assert points_3D.shape[0] == points_2D.shape[0], 'points 3D and points 2D must have same number of vertices'

    _, R_exp, t = cv2.solvePnP(points_3D,
                              np.ascontiguousarray(points_2D[:,:2]).reshape((-1,1,2)),
                              cameraMatrix,
                              distCoeffs)

    R, _ = cv2.Rodrigues(R_exp)
    return R, t

# Get the tightest bounding box surrounding keypoints
def get_2d_bb(box, size):
    x = box[0]
    y = box[1]
    min_x = np.min(np.reshape(box, [-1,2])[:,0])
    max_x = np.max(np.reshape(box, [-1,2])[:,0])
    min_y = np.min(np.reshape(box, [-1,2])[:,1])
    max_y = np.max(np.reshape(box, [-1,2])[:,1])
    w = max_x - min_x
    h = max_y - min_y
    new_box = [x*size, y*size, w*size, h*size]
    return new_box

# Compute IoU between two bounding boxes
def bbox_iou(box1, box2, x1y1x2y2=False):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0]-box1[2]/2.0, box2[0]-box2[2]/2.0)
        Mx = max(box1[0]+box1[2]/2.0, box2[0]+box2[2]/2.0)
        my = min(box1[1]-box1[3]/2.0, box2[1]-box2[3]/2.0)
        My = max(box1[1]+box1[3]/2.0, box2[1]+box2[3]/2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea/uarea

# Compute confidences of current keypoint predictions
def corner_confidences(gt_corners, pr_corners, th=80, sharpness=2, im_width=640, im_height=480):
    ''' gt_corners: Ground-truth 2D projections of the 3D bounding box corners, shape: (16 x nA), type: torch.FloatTensor
        pr_corners: Prediction for the 2D projections of the 3D bounding box corners, shape: (16 x nA), type: torch.FloatTensor
        th        : distance threshold, type: int
        sharpness : sharpness of the exponential that assigns a confidence value to the distance
        -----------
        return    : a torch.FloatTensor of shape (nA,) with 9 confidence values 
    '''
    shape = gt_corners.size()
    nA = shape[1]  
    dist = gt_corners - pr_corners
    num_el = dist.numel()
    num_keypoints = num_el//(nA*2)
    dist = dist.t().contiguous().view(nA, num_keypoints, 2)
    dist[:, :, 0] = dist[:, :, 0] * im_width
    dist[:, :, 1] = dist[:, :, 1] * im_height

    eps = 1e-5
    distthresh = torch.FloatTensor([th]).repeat(nA, num_keypoints) 
    dist = torch.sqrt(torch.sum((dist)**2, dim=2)).squeeze() # nA x 9
    mask = (dist < distthresh).type(torch.FloatTensor)
    conf = torch.exp(sharpness*(1 - dist/distthresh))-1  # mask * (torch.exp(math.log(2) * (1.0 - dist/rrt)) - 1)
    conf0 = torch.exp(sharpness*(1 - torch.zeros(conf.size(0),1))) - 1
    conf = conf / conf0.repeat(1, num_keypoints)
    # conf = 1 - dist/distthresh
    conf = mask * conf  # nA x 9
    mean_conf = torch.mean(conf, dim=1)
    return mean_conf

# Compute confidence of the current keypoint prediction
def corner_confidence(gt_corners, pr_corners, th=80, sharpness=2, im_width=640, im_height=480):
    ''' gt_corners: Ground-truth 2D projections of the 3D bounding box corners, shape: (18,) type: list
        pr_corners: Prediction for the 2D projections of the 3D bounding box corners, shape: (18,), type: list
        th        : distance threshold, type: int
        sharpness : sharpness of the exponential that assigns a confidence value to the distance
        -----------
        return    : a list of shape (9,) with 9 confidence values 
    '''
    dist = torch.FloatTensor(gt_corners) - pr_corners
    num_keypoints = dist.numel()//2
    dist = dist.view(num_keypoints, 2)
    dist[:, 0] = dist[:, 0] * im_width
    dist[:, 1] = dist[:, 1] * im_height
    eps = 1e-5
    dist  = torch.sqrt(torch.sum((dist)**2, dim=1))
    mask  = (dist < th).type(torch.FloatTensor)
    conf  = torch.exp(sharpness * (1.0 - dist/th)) - 1
    conf0 = torch.exp(torch.FloatTensor([sharpness])) - 1 + eps
    conf  = conf / conf0.repeat(num_keypoints, 1)
    # conf = 1.0 - dist/th
    conf  = mask * conf 
    return torch.mean(conf)

# Compute sigmoid
def sigmoid(x):
    return 1.0/(math.exp(-x)+1.)

# Compute softmax function
def softmax(x):
    x = torch.exp(x - torch.max(x))
    x = x/x.sum()
    return x

# Apply non-maxima suppression on a set of bounding boxes
def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes

    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = 1-boxes[i][4]                

    _,sortIds = torch.sort(det_confs)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            for j in range(i+1, len(boxes)):
                box_j = boxes[sortIds[j]]
                if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                    box_j[4] = 0
    return out_boxes

# Fix the wrong order of corners on the Occlusion dataset
def fix_corner_order(corners2D_gt):
    corners2D_gt_corrected = np.zeros((9, 2), dtype='float32')
    corners2D_gt_corrected[0, :] = corners2D_gt[0, :]
    corners2D_gt_corrected[1, :] = corners2D_gt[1, :]
    corners2D_gt_corrected[2, :] = corners2D_gt[3, :]
    corners2D_gt_corrected[3, :] = corners2D_gt[5, :]
    corners2D_gt_corrected[4, :] = corners2D_gt[7, :]
    corners2D_gt_corrected[5, :] = corners2D_gt[2, :]
    corners2D_gt_corrected[6, :] = corners2D_gt[4, :]
    corners2D_gt_corrected[7, :] = corners2D_gt[6, :]
    corners2D_gt_corrected[8, :] = corners2D_gt[8, :]
    return corners2D_gt_corrected

# Convert float tensors in GPU to tensors in CPU
def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

# Convert long tensors in GPU to tensors in CPU
def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)

# Get potential sets of predictions at test time
def get_multi_region_boxes(output, conf_thresh, num_classes, num_keypoints, anchors, num_anchors, correspondingclass, only_objectness=1, validation=False):
    
    # Parameters
    anchor_step = len(anchors)//num_anchors
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert(output.size(1) == (2*num_keypoints+1+num_classes)*num_anchors)
    h = output.size(2)
    w = output.size(3)

    # Activation
    t0 = time.time()
    all_boxes = []
    max_conf = -sys.maxsize
    max_cls_conf = -sys.maxsize
    output    = output.view(batch*num_anchors, 2*num_keypoints+1+num_classes, h*w).transpose(0,1).contiguous().view(2*num_keypoints+1+num_classes, batch*num_anchors*h*w)
    grid_x    = torch.linspace(0, w-1, w).repeat(h,1).repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).cuda()
    grid_y    = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).cuda()
    
    xs = list()
    ys = list()
    xs.append(torch.sigmoid(output[0]) + grid_x)
    ys.append(torch.sigmoid(output[1]) + grid_y)
    for j in range(1,num_keypoints):
        xs.append(output[2*j + 0] + grid_x)
        ys.append(output[2*j + 1] + grid_y)
    det_confs = torch.sigmoid(output[2*num_keypoints])
    cls_confs = torch.nn.Softmax()(Variable(output[2*num_keypoints+1:2*num_keypoints+1+num_classes].transpose(0,1))).data
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids   = cls_max_ids.view(-1)
    t1 = time.time()
    
    # GPU to CPU
    sz_hw = h*w
    sz_hwa = sz_hw*num_anchors
    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    for j in range(num_keypoints):
        xs[j] = convert2cpu(xs[j])
        ys[j] = convert2cpu(ys[j])
    if validation:
        cls_confs = convert2cpu(cls_confs.view(-1, num_classes))
    t2 = time.time()

    # Boxes filter
    for b in range(batch):
        boxes = []
        max_conf = -1
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b*sz_hwa + i*sz_hw + cy*w + cx
                    det_conf =  det_confs[ind]
                    if only_objectness:
                        conf = det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]
                    
                    if (det_confs[ind] > max_conf) and (cls_confs[ind, correspondingclass] > max_cls_conf):
                        max_conf = det_confs[ind]
                        max_cls_conf = cls_confs[ind, correspondingclass]
                        max_ind = ind                  
    
                    if conf > conf_thresh:
                        bcx = list()
                        bcy = list()
                        for j in range(num_keypoints):
                            bcx.append(xs[j][ind])
                            bcy.append(ys[j][ind])
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = list()
                        for j in range(num_keypoints):
                            box.append(bcx[j]/w)
                            box.append(bcy[j]/h)
                        box.append(det_conf)
                        box.append(cls_max_conf)
                        box.append(cls_max_id)    
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind]*tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
        if (len(boxes) == 0) or (not (correspondingclass in np.array(boxes)[:,2*num_keypoints+2])):
            bcx = list()
            bcy = list()
            for j in range(num_keypoints):
                bcx.append(xs[j][max_ind])
                bcy.append(ys[j][max_ind])
            cls_max_conf = max_cls_conf # cls_max_confs[max_ind]
            cls_max_id = correspondingclass # cls_max_ids[max_ind]
            det_conf = max_conf # det_confs[max_ind]
            box = list()
            for j in range(num_keypoints):
                box.append(bcx[j]/w)
                box.append(bcy[j]/h)
            box.append(det_conf)
            box.append(cls_max_conf)
            box.append(cls_max_id)     
            boxes.append(box)
            all_boxes.append(boxes)
        else:
            all_boxes.append(boxes)

    t3 = time.time()
    if False:
        print('---------------------------------')
        print('matrix computation : %f' % (t1-t0))
        print('        gpu to cpu : %f' % (t2-t1))
        print('      boxes filter : %f' % (t3-t2))
        print('---------------------------------')
    return all_boxes

# Read the labels from the file
def read_truths(lab_path, num_keypoints=9):
    num_labels = 2*num_keypoints+3 # +2 for width, height, +1 for class label
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(truths.size//num_labels, num_labels) # to avoid single truth problem
        return truths
    else:
        return np.array([])

def read_truths_args(lab_path, num_keypoints=9):
    num_labels = 2*num_keypoints+3
    truths = read_truths(lab_path)
    new_truths = []
    for i in range(truths.shape[0]):
        for j in range(num_labels):
            new_truths.append(truths[i][j])
    return np.array(new_truths)

def read_pose(lab_path):
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        return truths
    else:
        return np.array([])

def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names

def image2torch(img):
    width = img.width
    height = img.height
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    img = img.view(height, width, 3).transpose(0,1).transpose(0,2).contiguous()
    img = img.view(1, 3, height, width)
    img = img.float().div(255.0)
    return img

def read_data_cfg(datacfg):
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(datacfg, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        line = line.strip()
        if line == '':
            continue
        key,value = line.split('=')
        key = key.strip()
        value = value.strip()
        options[key] = value
    return options

def scale_bboxes(bboxes, width, height):
    import copy
    dets = copy.deepcopy(bboxes)
    for i in range(len(dets)):
        dets[i][0] = dets[i][0] * width
        dets[i][1] = dets[i][1] * height
        dets[i][2] = dets[i][2] * width
        dets[i][3] = dets[i][3] * height
    return dets
      
def file_lines(thefilepath):
    count = 0
    thefile = open(thefilepath, 'rb')
    while True:
        buffer = thefile.read(8192*1024)
        if not buffer:
            break
        count += buffer.count(b'\n')
    thefile.close( )
    return count

def get_image_size(fname):
    '''Determine the image type of fhandle and return its size.
    from draco'''
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24: 
            return
        if imghdr.what(fname) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return
            width, height = struct.unpack('>ii', head[16:24])
        elif imghdr.what(fname) == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif imghdr.what(fname) == 'jpeg' or imghdr.what(fname) == 'jpg':
            try:
                fhandle.seek(0) # Read 0xff next
                size = 2 
                ftype = 0 
                while not 0xc0 <= ftype <= 0xcf:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2 
                # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception: #IGNORE:W0703
                return
        else:
            return
        return width, height

def logging(message):
    print('%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message))
