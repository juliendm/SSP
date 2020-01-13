import sys
import time
from PIL import Image, ImageDraw
#from models.tiny_yolo import TinyYoloNet
from utils import *
from image import letterbox_image, correct_yolo_boxes
from darknet import Darknet

import pandas as pd
import numpy as np

import pickle
import cv2, json
from scipy.spatial.transform import Rotation

from scipy.optimize import fmin_bfgs

import json

namesfile=None
def detect(cfgfile, weightfile, imgfile):
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    with open('cfg/flips.json') as json_file:
        flipped = json.load(json_file)
    print('Loading Flipped... Done')

    # if m.num_classes == 20:
    #     namesfile = 'data/voc.names'
    # elif m.num_classes == 80:
    #     namesfile = 'data/coco.names'
    # else:
    #     namesfile = 'data/names'
    
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        m.cuda()

    class_names = load_class_names(namesfile)

    if imgfile == 'None':
        # baidu_data_submission = pd.read_csv('../../baidu_data/sample_submission.csv')
        # imgfiles = []
        # for img_id in baidu_data_submission['ImageId']:
        #     imgfiles.append('../../baidu_data/testing/images/%s.jpg' % img_id)

        with open('cfg/subm_pku_baidu.txt', 'r') as subm_file:
           imgfiles = subm_file.readlines()
        imgfiles = imgfiles

        # with open('cfg/subm_pku_baidu.txt', 'r') as subm_file:
        #    imgfiles = subm_file.readlines()

    else:
        imgfiles = [imgfile]
    print('Loading Image Files... Done')

    submission = {}

    for index,imgfile in enumerate(imgfiles):
        if index%50==0: print(index)

        imgpath = imgfile.rstrip()
        maskpath = imgpath.replace('images', 'masks').replace('JPEGImages', 'masks')
        segm_path = imgpath.replace('images', 'segmentations').replace('jpg', 'pkl')
        img_id = imgpath.split('/')[-1].split('.')[0]   
        yolov3_path = '../../baidu_data/predictions_yolov3/%s.pkl' % img_id

        imgpath_preprocessed = imgpath.replace('images', 'images_prepro')

        # try:
        #     img = Image.open(imgpath_preprocessed).convert('RGB')
        # except OSError:
        img = Image.open(imgpath).convert('RGB').crop((0,1497,3384,2710))
        try: #  it avoids the unnecessary call to os.path.exists()
            mask = Image.open(maskpath).convert('L').crop((0,1497,3384,2710))
            white_img = Image.new('RGB', img.size, (255, 255, 255))
            img = Image.composite(white_img, img, mask)
        except OSError:
            pass
            # img.save(imgpath_preprocessed)

        # with open(segm_path,'rb') as pkl_file:
        #     segms = pickle.load(pkl_file)

        # with open(yolov3_path,'rb') as pkl_file:
        #     boxes_yolov3 = pickle.load(pkl_file)

        if (img_id in flipped) and flipped[img_id]:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        sized = letterbox_image(img, m.width, m.height)

        # NN Predictions

        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        correct_yolo_boxes(boxes, img.width, img.height, m.width, m.height)
        finish = time.time()
        print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))


        # IOU boxes, boxes_yolov3

        boxes = np.array(boxes)
        boxes_mask = np.ones(boxes.shape[0], dtype=bool)
        filtered_boxes = []

        # print(len(boxes_yolov3))

        # for k in range(len(boxes_yolov3)):
        #     gt_boxes = torch.FloatTensor(boxes_yolov3[k])
        #     gt_boxes = gt_boxes.repeat(len(boxes[boxes_mask]),1).t()
        #     pred_boxes = torch.FloatTensor(boxes[boxes_mask]).t()
        #     best_iou, best_i = torch.max(multi_bbox_ious(gt_boxes, pred_boxes, x1y1x2y2=False),0)
        #     if best_iou > 0.1:
        #         filtered_boxes.append(boxes[boxes_mask][best_i])
        #     boxes_mask[best_i] = 0
        # filtered_boxes = np.array(filtered_boxes)

        # print(len(filtered_boxes))

        filtered_boxes = boxes


            # # pred_boxes and gt_boxes are transposed for torch.max
            # if best_iou > iou_thresh and pred_boxes[6][best_j] == gt_boxes[6][0]:
            #     correct += 1


        # ious = multi_bbox_ious( torch.FloatTensor(boxes).t(),  torch.FloatTensor(boxes_yolov3).t(), x1y1x2y2=False)

        # print(ious)

        # Process Predictions

        vertices_2D = []
        vertices_2D_colored = []
        triangles_2D = []

        pred_candidates = []

        num_keypoints = 10

        for i in range(len(filtered_boxes)):
            box = filtered_boxes[i]

            if (img_id in flipped) and flipped[img_id]:
                box[0] =  1.0 - box[0]

                corners = np.array(box[7:7+2*num_keypoints]).reshape(9,2)
                corners[:,0] = 1.0 - corners[:,0]

                corners[1+0,0],corners[1+4,0] = corners[1+4,0],corners[1+0,0]
                corners[1+0,1],corners[1+4,1] = corners[1+4,1],corners[1+0,1]

                corners[1+1,0],corners[1+5,0] = corners[1+5,0],corners[1+1,0]
                corners[1+1,1],corners[1+5,1] = corners[1+5,1],corners[1+1,1]

                corners[1+2,0],corners[1+6,0] = corners[1+6,0],corners[1+2,0]
                corners[1+2,1],corners[1+6,1] = corners[1+6,1],corners[1+2,1]

                corners[1+3,0],corners[1+7,0] = corners[1+7,0],corners[1+3,0]
                corners[1+3,1],corners[1+7,1] = corners[1+7,1],corners[1+3,1]

                for j in range(num_keypoints-1):
                    box[2*j+7] = corners[j,0]
                    box[2*j+1+7] = corners[j,1]

            cls_id = int(box[6])
            model_id = car_class2id[cls_id]

            with open('../../baidu_data/models/json/%s.json' % car_id2name[model_id]) as json_file:
                data = json.load(json_file)

            vertices  = np.c_[np.array(data['vertices']), np.ones((len(data['vertices']), 1))].transpose()
            triangles = np.array(data['faces'])-1
            corners3D     = get_3D_corners(vertices,mean=True)   
            objpoints3D = np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32')

            K = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)

            # Denormalize the corner predictions 
            corners2D = np.array(box[7:7+2*(num_keypoints-1)], dtype='float32').reshape(num_keypoints-1,2)
            corners2D[:, 0] = corners2D[:, 0] * 3384.0
            corners2D[:, 1] = corners2D[:, 1] * (2710.0-1497.0) + 1497.0

            R_pr, t_pr = pnp(objpoints3D,  corners2D, K)
            Rt_pr      = np.concatenate((R_pr, t_pr), axis=1)

            angles = Rotation.from_dcm(R_pr.T).as_euler('xyz')

            # Make Prediction Candidates
            pred_candidate = [angles[0],angles[1],angles[2],t_pr[0,0],t_pr[1,0],t_pr[2,0],box[5],model_id]
            pred_candidates.append(pred_candidate)
            

        #     # if i == 0:
        #     #     angles = Rotation.from_dcm(R_pr.T).as_euler('xyz')
        #     #     # 
                
        #     #     with open('mask.pkl','rb') as pkl_file:
        #     #         mask = pickle.load(pkl_file)

        #     #     x0 = [t_pr[0,0],t_pr[1,0],t_pr[2,0],angles[0],angles[1],angles[2]]

        #     #     # Optim:
        #     #     #res = fmin_bfgs(neg_iou_mask, x0, args=(coords,mask),epsilon=1e-03,disp=1)
        #     #     x_sol = [-5.00803688 ,2.9252357 ,12.67238289 ,0.15001176 ,-0.0128817 , -0.04438048]

        #     #     iou = -neg_iou_mask(x0,coords,mask,save=True)
        #     #     print(iou)

        #     #     R_pr = Rotation.from_euler('xyz', x_sol[3:]).as_dcm().T
        #     #     Rt_pr = np.concatenate((R_pr, np.array(x_sol[:3]).reshape(-1,1)), axis=1)

            proj_corners2D  = np.transpose(compute_projection(corners3D, Rt_pr, K))
            proj_corners2D[:, 0] = proj_corners2D[:, 0] / 3384.0
            proj_corners2D[:, 1] = (proj_corners2D[:, 1] - 1497.0) / (2710.0-1497.0) 
            
            for j in range(num_keypoints-2):
                filtered_boxes[i][9+2*j]   = proj_corners2D[j, 0]
                filtered_boxes[i][9+2*j+1] = proj_corners2D[j, 1]

            vertices_proj_2d = np.transpose(compute_projection(vertices, Rt_pr, K))
            vertices_proj_2d[:, 0] = vertices_proj_2d[:, 0] / 3384.0
            vertices_proj_2d[:, 1] = (vertices_proj_2d[:, 1] - 1497.0) / (2710.0-1497.0) 
            vertices_2D.append(vertices_proj_2d)

            # vertices_colored =  np.c_[coords[:,:3], np.ones((len(data['vertices']), 1))].transpose()
            # vertices_proj_2d_colored = np.transpose(compute_projection(vertices_colored, Rt_pr, K))
            # vertices_proj_2d_colored = np.c_[vertices_proj_2d_colored, coords[:,3]]
            # vertices_proj_2d_colored[:, 0] = vertices_proj_2d_colored[:, 0] / 3384.0
            # vertices_proj_2d_colored[:, 1] = (vertices_proj_2d_colored[:, 1] - 1497.0) / (2710.0-1497.0) 
            # vertices_2D_colored.append(vertices_proj_2d_colored)

            triangles_2D.append(triangles)


        if (img_id in flipped) and flipped[img_id]:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        plot_boxes(img, filtered_boxes, '../../baidu_data/predictions/%s.jpg' % img_id, class_names, vertices_2D, triangles_2D)
        #plot_boxes(img, boxes, '../../baidu_data/predictions/%s.jpg' % img_id, class_names, vertices_2D_colored)

        pred_str = ''

        # # Filter With IOU
        # pred_candidates = np.array(pred_candidates)
        # preds_mask = np.ones(pred_candidates.shape[0], dtype=bool)
        # for segm_idx in range(len(segms)):
        #     match_idx = 0
        #     max_iou = 0.0
        #     for pred_idx in range(len(pred_candidates[preds_mask])):
        #         pred = pred_candidates[preds_mask][pred_idx]
        #         x0 = [pred[3],pred[4],pred[5],pred[0],pred[1],pred[2]]
        #         model_id = pred[7]
        #         with open('../../baidu_data/models/pkl/%s.pkl' % car_id2name[model_id], 'rb') as pkl_file:
        #             coords = pickle.load(pkl_file)
        #         coords[(coords[:,3]==0)&(coords[:,1]<-0.3),3]=7

        #         iou = -neg_iou_mask(x0,coords,segms[segm_idx])
        #         if iou > max_iou:
        #             max_iou = iou
        #             match_idx = pred_idx
        #     if max_iou == 0.0:
        #         continue
        #     print(max_iou)
        #     pred = pred_candidates[preds_mask][match_idx]
        #     print(pred)
        #     pred_str += '%f %f %f %f %f %f %f ' % (pred[0],pred[1],-pred[2]+np.pi,pred[3],pred[4],pred[5],pred[6])
        #     preds_mask[pred_idx] = 0

        # Filter With Score
        pred_candidates = np.array(pred_candidates)
        for pred_idx in range(len(pred_candidates)):
            pred = pred_candidates[pred_idx]
            pred_str += '%f %f %f %f %f %f %f ' % (pred[0],pred[1],-pred[2]+np.pi,pred[3],pred[4],pred[5],pred[6])

        # Save
        submission[img_id] = pred_str.rstrip()



    print('Saving File')
    submission_array = []
    for key, value in submission.items():
        submission_array.append([key,value])
    df_submission = pd.DataFrame(submission_array,columns=['ImageId','PredictionString'])
    df_submission.to_csv('submission.csv',index=False)
    print('Done')



def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    
    use_cuda = True
    if use_cuda:
        m.cuda()

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)

def detect_skimage(cfgfile, weightfile, imgfile):
    from skimage import io
    from skimage.transform import resize
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    
    use_cuda = True
    if use_cuda:
        m.cuda()

    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255
    
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)

if __name__ == '__main__':
    if len(sys.argv) == 5:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        imgfile = sys.argv[3]
        globals()["namesfile"] = sys.argv[4]
        detect(cfgfile, weightfile, imgfile)
        #detect_cv2(cfgfile, weightfile, imgfile)
        #detect_skimage(cfgfile, weightfile, imgfile)
    else:
        print('Usage: ')
        print('  python detect.py cfgfile weightfile imgfile names')
        #detect('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights', 'data/person.jpg', version=1)
