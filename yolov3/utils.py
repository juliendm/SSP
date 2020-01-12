import sys
import os
import time
import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import itertools
import struct # get_image_size
import imghdr # get_image_size

from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cv2, json
from scipy.spatial.transform import Rotation

car_name2id = {'019-SUV': 46, '036-CAR01': 47, '037-CAR02': 16, 'MG-GT-2015': 30, 'Skoda_Fabia-2011': 67, 'aodi-Q7-SUV': 48, 'aodi-a6': 17, 'baojun-310-2017': 0, 'baojun-510': 49, 'baoma-330': 18, 'baoma-530': 19, 'baoma-X5': 50, 'baoshijie-kayan': 51, 'baoshijie-paoche': 20, 'beiqi-huansu-H3': 52, 'benchi-GLK-300': 53, 'benchi-ML500': 54, 'benchi-SUR': 71, 'bentian-fengfan': 21, 'biaozhi-3008': 1, 'biaozhi-408': 22, 'biaozhi-508': 23, 'biaozhi-liangxiang': 2, 'bieke': 37, 'bieke-kaiyue': 24, 'bieke-yinglang-XT': 3, 'biyadi-2x-F0': 4, 'biyadi-F3': 38, 'biyadi-qin': 39, 'biyadi-tang': 72, 'changan-CS35-2012': 73, 'changan-cs5': 74, 'changanbenben': 5, 'changcheng-H6-2016': 75, 'dazhong': 40, 'dazhong-SUV': 76, 'dazhongmaiteng': 41, 'dihao-EV': 42, 'dongfeng-DS5': 6, 'dongfeng-fengguang-S560': 77, 'dongfeng-fengxing-SX6': 78, 'dongfeng-xuetielong-C6': 43, 'dongfeng-yulong-naruijie': 45, 'dongnan-V3-lingyue-2011': 44, 'feiyate': 7, 'fengtian-MPV': 9, 'fengtian-SUV-gai': 56, 'fengtian-liangxiang': 8, 'fengtian-puladuo-06': 55, 'fengtian-weichi-2006': 15, 'fute': 25, 'guangqi-chuanqi-GS4-2015': 57, 'haima-3': 26, 'jianghuai-ruifeng-S3': 58, 'jili-boyue': 59, 'jilixiongmao-2015': 10, 'jipu-3': 60, 'kaidilake-CTS': 27, 'leikesasi': 28, 'lingmu-SX4-2012': 13, 'lingmu-aotuo-2009': 11, 'lingmu-swift': 12, 'linken-SUV': 61, 'lufeng-X8': 62, 'mazida-6-2015': 29, 'oubao': 31, 'qirui-ruihu': 63, 'qiya': 32, 'rongwei-750': 33, 'rongwei-RX5': 64, 'sanling-oulande': 65, 'sikeda-SUV': 66, 'sikeda-jingrui': 14, 'supai-2016': 34, 'xiandai-i25-2016': 68, 'xiandai-suonata': 35, 'yingfeinidi-SUV': 70, 'yingfeinidi-qx80': 69, 'yiqi-benteng-b50': 36}
car_id2name = {v: k for k, v in car_name2id.items()}
car_id2class = {2:0, 6:1, 7:2, 8:3, 9:4, 12:5, 14:6, 16:7, 18:8, 19:9, 20:10, 23:11, 25:12, 27:13, 28:14, 31:15, 32:16, 35:17, 37:18, 40:19, 43:20, 46:21, 47:22, 48:23, 50:24, 51:25, 54:26, 56:27, 60:28, 61:29, 66:30, 70:31, 71:32, 76:33}
car_class2id = {v: k for k, v in car_id2class.items()}

def sigmoid(x):
    return 1.0/(math.exp(-x)+1.)

def softmax(x):
    x = torch.exp(x - torch.max(x))
    x = x/x.sum()
    return x

def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        x1_min = min(box1[0], box2[0])
        x2_max = max(box1[2], box2[2])
        y1_min = min(box1[1], box2[1])
        y2_max = max(box1[3], box2[3])
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    else:
        w1, h1 = box1[2], box1[3]
        w2, h2 = box2[2], box2[3]
        x1_min = min(box1[0]-w1/2.0, box2[0]-w2/2.0)
        x2_max = max(box1[0]+w1/2.0, box2[0]+w2/2.0)
        y1_min = min(box1[1]-h1/2.0, box2[1]-h2/2.0)
        y2_max = max(box1[1]+h1/2.0, box2[1]+h2/2.0)

    w_union = x2_max - x1_min
    h_union = y2_max - y1_min
    w_cross = w1 + w2 - w_union
    h_cross = h1 + h2 - h_union
    carea = 0
    if w_cross <= 0 or h_cross <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = w_cross * h_cross
    uarea = area1 + area2 - carea
    return float(carea/uarea)

def multi_bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        x1_min = torch.min(boxes1[0], boxes2[0])
        x2_max = torch.max(boxes1[2], boxes2[2])
        y1_min = torch.min(boxes1[1], boxes2[1])
        y2_max = torch.max(boxes1[3], boxes2[3])
        w1, h1 = boxes1[2] - boxes1[0], boxes1[3] - boxes1[1]
        w2, h2 = boxes2[2] - boxes2[0], boxes2[3] - boxes2[1]
    else:
        w1, h1 = boxes1[2], boxes1[3]
        w2, h2 = boxes2[2], boxes2[3]
        x1_min = torch.min(boxes1[0]-w1/2.0, boxes2[0]-w2/2.0)
        x2_max = torch.max(boxes1[0]+w1/2.0, boxes2[0]+w2/2.0)
        y1_min = torch.min(boxes1[1]-h1/2.0, boxes2[1]-h2/2.0)
        y2_max = torch.max(boxes1[1]+h1/2.0, boxes2[1]+h2/2.0)

    w_union = x2_max - x1_min
    h_union = y2_max - y1_min
    w_cross = w1 + w2 - w_union
    h_cross = h1 + h2 - h_union
    mask = (((w_cross <= 0) + (h_cross <= 0)) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = w_cross * h_cross
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea/uarea

def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes

    res =[]
    for item in boxes:
        temp = []
        for ite in item:
            if torch.is_tensor(ite):
                ite = float(ite.numpy())
            temp.append(ite)
        res.append(temp)
    boxes = res

    det_confs = np.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = 1-boxes[i][4]

    sortIds = np.argsort(det_confs)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            for j in range(i+1, len(boxes)):
                box_j = boxes[sortIds[j]]
                if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                    #print(box_i, box_j, bbox_iou(box_i, box_j, x1y1x2y2=False))
                    box_j[4] = 0
    return out_boxes

def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)

def get_all_boxes(output, netshape, conf_thresh, num_classes, only_objectness=1, validation=False, use_cuda=True):
    # total number of inputs (batch size)
    # first element (x) for first tuple (x, anchor_mask, num_anchor)
    tot = output[0]['x'].data.size(0)
    all_boxes = [[] for i in range(tot)]
    for i in range(len(output)):
        pred = output[i]['x'].data

        # find number of workers (.s.t, number of GPUS) 
        nw = output[i]['n'].data.size(0)
        anchors = output[i]['a'].chunk(nw)[0]
        num_anchors = output[i]['n'].data[0].item()

        b = get_region_boxes(pred, netshape, conf_thresh, num_classes, anchors, num_anchors, \
                only_objectness=only_objectness, validation=validation, use_cuda=use_cuda)
        for t in range(tot):
            all_boxes[t] += b[t]
    return all_boxes

def get_region_boxes(output, netshape, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1, validation=False, use_cuda=True):

    num_keypoints = 1
    num_labels = 12

    device = torch.device("cuda" if use_cuda else "cpu")
    anchors = anchors.to(device)
    anchor_step = anchors.size(0)//num_anchors
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert(output.size(1) == (num_labels+num_classes)*num_anchors)
    h = output.size(2)
    w = output.size(3)
    cls_anchor_dim = batch*num_anchors*h*w
    if netshape[0] != 0:
        nw, nh = netshape
    else:
        nw, nh = w, h

    t0 = time.time()
    all_boxes = []
    output = output.view(batch*num_anchors, num_labels+num_classes, h*w).transpose(0,1).contiguous().view(num_labels+num_classes, cls_anchor_dim)

    grid_x = torch.linspace(0, w-1, w).repeat(batch*num_anchors, h, 1).view(cls_anchor_dim).to(device)
    grid_y = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(batch*num_anchors, 1, 1).view(cls_anchor_dim).to(device)
    ix = torch.LongTensor(range(0,2)).to(device)
    anchor_w = anchors.view(num_anchors, anchor_step).index_select(1, ix[0]).repeat(batch, h*w).view(cls_anchor_dim)
    anchor_h = anchors.view(num_anchors, anchor_step).index_select(1, ix[1]).repeat(batch, h*w).view(cls_anchor_dim)

    # HAS TO BE IMPROVED
    xs, ys = output[0].sigmoid() + grid_x, output[1].sigmoid() + grid_y

    if num_keypoints > 1:
        corners = output[2:2*num_keypoints].view((num_keypoints-1),2,-1)
        xcs, ycs = corners[:,0] + grid_x, corners[:,1] + grid_y

    trans_rot = output[2*num_keypoints:2*num_keypoints+7]

    ws, hs = output[2*num_keypoints+7].exp() * anchor_w.detach(), output[2*num_keypoints+7+1].exp() * anchor_h.detach()
    det_confs = output[num_labels-1].sigmoid()

    # by ysyun, dim=1 means input is 2D or even dimension else dim=0
    cls_confs = torch.nn.Softmax(dim=1)(output[num_labels:num_labels+num_classes].transpose(0,1)).detach()
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)
    t1 = time.time()
    
    sz_hw = h*w
    sz_hwa = sz_hw*num_anchors
    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs, ys = convert2cpu(xs), convert2cpu(ys)
    if num_keypoints > 1:
        xcs, ycs = convert2cpu(xcs), convert2cpu(ycs)
    trans_rot = convert2cpu(trans_rot)
    ws, hs = convert2cpu(ws), convert2cpu(hs)
    if validation:
        cls_confs = convert2cpu(cls_confs.view(-1, num_classes))

    t2 = time.time()
    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b*sz_hwa + i*sz_hw + cy*w + cx
                    det_conf =  det_confs[ind]
                    conf = det_conf * (cls_max_confs[ind] if not only_objectness else 1.0)
    
                    if conf > conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx/w, bcy/h, bw/nw, bh/nh, det_conf, cls_max_conf, cls_max_id]
                        for cor in range(num_keypoints-1):
                            box.append(xcs[cor][ind]/w)
                            box.append(ycs[cor][ind]/h)
                        for tr_idx in range(7):
                            box.append(trans_rot[tr_idx][ind])
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind]*tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
        all_boxes.append(boxes)
    t3 = time.time()
    if False:
        print('---------------------------------')
        print('matrix computation : %f' % (t1-t0))
        print('        gpu to cpu : %f' % (t2-t1))
        print('      boxes filter : %f' % (t3-t2))
        print('---------------------------------')
    return all_boxes

def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None):
    import cv2
    colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]])
    def get_color(c, x, max_val):
        ratio = float(x)/max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
        return int(r*255)

    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(round((box[0] - box[2]/2.0) * width))
        y1 = int(round((box[1] - box[3]/2.0) * height))
        x2 = int(round((box[0] + box[2]/2.0) * width))
        y2 = int(round((box[1] + box[3]/2.0) * height))

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            #print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red   = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue  = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            img = cv2.putText(img, class_names[cls_id], (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
        img = cv2.rectangle(img, (x1,y1), (x2,y2), rgb, 1)
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    return img

def drawrect(drawcontext, xy, outline=None, width=0):
    x1, y1, x2, y2 = xy
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill=outline, width=width)

def drawbox(drawcontext, xs, ys, outline=None, width=0):
    edges_corners = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]                    
    for edge in edges_corners:
        points = (xs[edge[0]], ys[edge[0]]), (xs[edge[1]], ys[edge[1]])
        drawcontext.line(points, fill=outline, width=width)

def drawmesh(drawcontext, vertices, triangles, im_width, im_height, outline=None, width=0):
    for tri in triangles:
        points = (vertices[tri[0],0]*im_width, vertices[tri[0],1]*im_height), \
                 (vertices[tri[1],0]*im_width, vertices[tri[1],1]*im_height), \
                 (vertices[tri[2],0]*im_width, vertices[tri[2],1]*im_height), \
                 (vertices[tri[0],0]*im_width, vertices[tri[0],1]*im_height)
        drawcontext.line(points, fill=outline, width=width)

def drawhull(drawcontext, vertices, im_width, im_height, outline=None, width=0):

    hull = ConvexHull(vertices)
    ver = hull.vertices

    for i in range(len(ver)-1):
        points = (vertices[ver[i],0]*im_width, vertices[ver[i],1]*im_height), \
                 (vertices[ver[i+1],0]*im_width, vertices[ver[i+1],1]*im_height)
        drawcontext.line(points, fill=outline, width=width)

def neg_iou_mask(x,coords,mask,save=False):

    # print(x)

    # delta_x,delta_y,delta_z,angle_x,angle_y,angle_z

    K = np.array([[2304.5479, 0,  1686.2379],
                  [0, 2305.8757, 1354.9849],
                  [0, 0, 1]], dtype=np.float32)

    R_pr = Rotation.from_euler('xyz', x[3:]).as_dcm().T
    Rt_pr = np.concatenate((R_pr, np.array(x[:3]).reshape(-1,1)), axis=1)

    vertices_colored =  np.c_[coords[:,:3], np.ones((len(coords), 1))].transpose()
    vertices_proj_2d_colored = np.transpose(compute_projection(vertices_colored, Rt_pr, K))
    vertices_proj_2d_colored = np.c_[vertices_proj_2d_colored, coords[:,3]]
    vertices_proj_2d_colored[:, 0] = vertices_proj_2d_colored[:, 0] / 3384.0
    vertices_proj_2d_colored[:, 1] = (vertices_proj_2d_colored[:, 1] - 1497.0) / (2710.0-1497.0) 

    mask_pr = np.zeros(mask.shape,dtype=int)
    
    # drawmask(mask_pr, vertices_proj_2d, triangles, mask.shape[1], mask.shape[0])
    # Much Faster (Approximation)
    drawmaskhull(mask_pr, vertices_proj_2d_colored, mask.shape[1], mask.shape[0])

    if save:
        plt.imsave('mask_pr.png', mask_pr, cmap=cm.gray)

    iou = np.sum(mask&mask_pr)/np.sum(mask|mask_pr)

    # print(iou)

    return -iou

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

def compute_projection(points_3D, transformation, internal_calibration):
    projections_2d = np.zeros((2, points_3D.shape[1]), dtype='float32')
    camera_projection = (internal_calibration.dot(transformation)).dot(points_3D)
    projections_2d[0, :] = camera_projection[0, :]/camera_projection[2, :]
    projections_2d[1, :] = camera_projection[1, :]/camera_projection[2, :]
    return projections_2d

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

def drawmaskhull(mask, coords, im_width, im_height):
    for index in range(8):
        vertices = coords[coords[:,2]==index,:2]
        if len(vertices):
            hull = ConvexHull(vertices)
            ver = hull.vertices

            xs = vertices[ver,0]*im_width
            ys = vertices[ver,1]*im_height

            ind_x,ind_y = inside_polygone(ys,xs)

            idx_mask = np.ones(ind_x.shape[0], dtype=bool)
            idx_mask = np.logical_and(idx_mask, ind_x >= 0)
            idx_mask = np.logical_and(idx_mask, ind_x < mask.shape[0])
            idx_mask = np.logical_and(idx_mask, ind_y >= 0)
            idx_mask = np.logical_and(idx_mask, ind_y < mask.shape[1])

            mask[ind_x[idx_mask],ind_y[idx_mask]] = 1

def drawmask(mask, vertices, triangles, im_width, im_height):
    for tri in triangles:
        x0 = int(vertices[tri[0],0]*im_width)
        y0 = int(vertices[tri[0],1]*im_height)
        x1 = int(vertices[tri[1],0]*im_width)
        y1 = int(vertices[tri[1],1]*im_height)
        x2 = int(vertices[tri[2],0]*im_width)
        y2 = int(vertices[tri[2],1]*im_height)

        xs=np.array((x0,x1,x2),dtype=float)
        ys=np.array((y0,y1,y2),dtype=float)
        ind_x,ind_y = inside_polygone(ys,xs)
        mask[ind_x,ind_y] = 1

def inside_polygone(xs,ys):

    x_range=np.arange(np.min(xs),np.max(xs)+1)
    y_range=np.arange(np.min(ys),np.max(ys)+1)

    X,Y=np.meshgrid(x_range,y_range)
    xc=np.mean(xs)
    yc=np.mean(ys)

    mask = np.ones(X.shape,dtype=bool)

    for i in range(len(xs)):
        ii=(i+1)%len(xs)
        if xs[i]==xs[ii]:
            include = X *(xc-xs[i])/abs(xc-xs[i]) >= xs[i] *(xc-xs[i])/abs(xc-xs[i])
        else:
            poly=np.poly1d([(ys[ii]-ys[i])/(xs[ii]-xs[i]),ys[i]-xs[i]*(ys[ii]-ys[i])/(xs[ii]-xs[i])])
            include = Y *(yc-poly(xc))/abs(yc-poly(xc)) >= poly(X) *(yc-poly(xc))/abs(yc-poly(xc))
        mask *= include

    return X[mask].astype(int), Y[mask].astype(int)


def drawtext(img, pos, text, bgcolor=(255,255,255), font=None):
    if font is None:
        font = ImageFont.load_default().font
    (tw, th) = font.getsize(text)
    box_img = Image.new('RGB', (tw+2, th+2), bgcolor)
    ImageDraw.Draw(box_img).text((0, 0), text, fill=(0,0,0,255), font=font)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    sx, sy = int(pos[0]),int(pos[1]-th-2)
    if sx<0:
        sx=0
    if sy<0:
        sy=0
    img.paste(box_img, (sx, sy))

def plot_boxes(img, boxes, savename=None, class_names=None, vertices_2D=None, triangles_2D=None):
    num_keypoints = 1
    num_labels = 12
    colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]])
    def get_color(c, x, max_val):
        ratio = float(x)/max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
        return int(r*255)

    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arialbd", 14)
    except:
        font=None
    print("%d box(es) is(are) found" % len(boxes))

    mask = np.zeros((height,width),dtype=int)
    for i in range(len(boxes)):
        box = boxes[i]
        x1,y1,x2,y2 = (box[0] - box[2]/2.0) * width, (box[1] - box[3]/2.0) * height, \
                (box[0] + box[2]/2.0) * width, (box[1] + box[3]/2.0) * height

        rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = int(box[6])
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red   = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue  = get_color(0, offset, classes)
            rgb = (red, green, blue)
            text = "{} : {:.3f}".format(class_names[cls_id],cls_conf)
            drawtext(img, (x1, y1), text, bgcolor=rgb, font=font)
        drawrect(draw, [x1, y1, x2, y2], outline=rgb, width=2)
        #corners = np.array(box[9:9+2*(num_keypoints-1)]).reshape(8,2)
        #drawbox(draw, corners[:,0]*width, corners[:,1]*height, outline=rgb, width=2)
        # if vertices_2D is not None and triangles_2D is not None:
            # drawmesh(draw, vertices_2D[i],  triangles_2D[i], width, height, outline=rgb, width=1)
            # drawhull(draw, vertices_2D[i], width, height, outline=rgb, width=1)
            # drawmask(mask, vertices_2D[i],  triangles_2D[i], width, height)
        # if vertices_2D is not None and triangles_2D is None:
        #     drawmaskhull(mask, vertices_2D[i], width, height)
    if savename:
        print("save plot results to %s" % savename)
        img.save(savename)
        # img = Image.fromarray(mask,'L')
        # img.save('mask.jpg')
        # plt.imsave('mask.png', mask, cmap=cm.gray)
    return img

def read_truths(lab_path):
    if not os.path.exists(lab_path):
        return np.array([])
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        num_keypoints = 1
        num_labels = 12
        truths = truths.reshape(-1, num_labels) # to avoid single truth problem
        return truths
    else:
        return np.array([])

def read_truths_args(lab_path, min_box_scale):
    num_keypoints = 1
    num_labels = 12
    truths = read_truths(lab_path)
    new_truths = []
    for i in range(truths.shape[0]):

        new_truths.append(car_id2class[truths[i][0]])
        for j in range(num_keypoints):
            new_truths.append(truths[i][2*j+1])
            new_truths.append((truths[i][2*j+2]*2710.0-1497.0)/(2710.0-1497.0))
        new_truths.append(truths[i][-2])
        new_truths.append(truths[i][-1]*2710.0/(2710.0-1497.0))


        # truths[i][2] = (truths[i][2]*2710.0-1497.0)/(2710.0-1497.0)
        # truths[i][20] = truths[i][20]*2710.0/(2710.0-1497.0)
        # truths[i][0] = car_id2class[truths[i][0]]
        # new_truths.append([truths[i][0], truths[i][1], truths[i][2], truths[i][19], truths[i][20]])


    return np.array(new_truths)

def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r', encoding='utf8') as fp:
        lines = fp.readlines()
    for line in lines:
        class_names.append(line.strip())
    return class_names

def image2torch(img):
    if isinstance(img, Image.Image):
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(height, width, 3).transpose(0,1).transpose(0,2).contiguous()
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)
    elif type(img) == np.ndarray: # cv2 image
        img = torch.from_numpy(img.transpose(2,0,1)).float().div(255.0).unsqueeze(0)
    else:
        print("unknown image type")
        exit(-1)
    return img

import types
def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=True):
    model.eval()
    t0 = time.time()
    img = image2torch(img)
    t1 = time.time()

    img = img.to(torch.device("cuda" if use_cuda else "cpu"))
    t2 = time.time()

    out_boxes = model(img)
    if model.net_name() == 'region': # region_layer
        shape=(0,0)
    else:
        shape=(model.width, model.height)
    boxes = get_all_boxes(out_boxes, shape, conf_thresh, model.num_classes, use_cuda=use_cuda)[0]
    
    t3 = time.time()
    boxes = nms(boxes, nms_thresh)
    t4 = time.time()

    if False:
        print('-----------------------------------')
        print(' image to tensor : %f' % (t1 - t0))
        print('  tensor to cuda : %f' % (t2 - t1))
        print('         predict : %f' % (t3 - t2))
        print('             nms : %f' % (t4 - t3))
        print('           total : %f' % (t4 - t0))
        print('-----------------------------------')
    return boxes

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

def savelog(message):
    logging(message)
    with open('savelog.txt', 'a') as f:
        print('%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message), file=f)
