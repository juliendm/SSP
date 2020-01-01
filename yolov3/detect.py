import sys
import time
from PIL import Image, ImageDraw
#from models.tiny_yolo import TinyYoloNet
from utils import *
from image import letterbox_image, correct_yolo_boxes
from darknet import Darknet

import pickle
import cv2, json
from scipy.spatial.transform import Rotation

from scipy.optimize import fmin_bfgs

namesfile=None
def detect(cfgfile, weightfile, imgfile):
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    # if m.num_classes == 20:
    #     namesfile = 'data/voc.names'
    # elif m.num_classes == 80:
    #     namesfile = 'data/coco.names'
    # else:
    #     namesfile = 'data/names'
    
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        m.cuda()

    img = Image.open(imgfile).convert('RGB').crop((0,1497,3384,2710))
    sized = letterbox_image(img, m.width, m.height)

    start = time.time()
    boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
    correct_yolo_boxes(boxes, img.width, img.height, m.width, m.height)

    finish = time.time()
    print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))


    vertices_2D = []
    triangles_2D = []
    for i in range(len(boxes)):
        box = boxes[i]

        cls_id = int(box[6])
        model_id = car_class2id[cls_id]

        with open('../../baidu_data/models/json/%s.json' % car_id2name[model_id]) as json_file:
            data = json.load(json_file)
        with open('../../baidu_data/models/pkl/%s.pkl' % car_id2name[model_id], 'rb') as pkl_file:
            coords = pickle.load(pkl_file)
        vertices  = np.c_[np.array(data['vertices']), np.ones((len(data['vertices']), 1))].transpose()
        # triangles = np.array(data['faces'])-1
        corners3D     = get_3D_corners(vertices)   
        objpoints3D = np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32')

        K = np.array([[2304.5479, 0,  1686.2379],
                      [0, 2305.8757, 1354.9849],
                      [0, 0, 1]], dtype=np.float32)
        num_keypoints = 10

        # Denormalize the corner predictions 
        corners2D = np.array(box[7:7+2*(num_keypoints-1)], dtype='float32').reshape(num_keypoints-1,2)
        corners2D[:, 0] = corners2D[:, 0] * 3384.0
        corners2D[:, 1] = corners2D[:, 1] * (2710.0-1497.0) + 1497.0

        R_pr, t_pr = pnp(objpoints3D,  corners2D, K)
        Rt_pr           = np.concatenate((R_pr, t_pr), axis=1)
        proj_corners2D  = np.transpose(compute_projection(corners3D, Rt_pr, K))


        # vertices_proj_2d = np.transpose(compute_projection(vertices, Rt_pr, K))

        vertices_colored =  np.c_[coords[:,:3], np.ones((len(data['vertices']), 1))].transpose()
        vertices_proj_2d_colored = np.transpose(compute_projection(vertices_colored, Rt_pr, K))
        vertices_proj_2d_colored = np.c_[vertices_proj_2d_colored, coords[:,3]]


        if i == 0:
            angles = Rotation.from_dcm(R_pr.T).as_euler('xyz')
            # R_pr = Rotation.from_euler('xyz', angles).as_dcm().T
            
            mask = np.zeros((2710-1497,3384),dtype=int)
            for a in range(300,600):
                for b in range(500,900):
                    mask[a,b] = 1
            plt.imsave('mask_ref.png', mask, cmap=cm.gray)

            x0 = [t_pr[0,0],t_pr[1,0],t_pr[2,0],angles[0],angles[1],angles[2]]
            
            iou = -neg_iou_mask(x0,coords,mask)
            #res = fmin_bfgs(neg_iou_mask, x0, args=(coords,mask),epsilon=1e-03,disp=1)


        proj_corners2D[:, 0] = proj_corners2D[:, 0] / 3384.0
        proj_corners2D[:, 1] = (proj_corners2D[:, 1] - 1497.0) / (2710.0-1497.0) 

        vertices_proj_2d_colored[:, 0] = vertices_proj_2d_colored[:, 0] / 3384.0
        vertices_proj_2d_colored[:, 1] = (vertices_proj_2d_colored[:, 1] - 1497.0) / (2710.0-1497.0) 

        for j in range(num_keypoints-2):
            boxes[i][9+2*j]   = proj_corners2D[j, 0]
            boxes[i][9+2*j+1] = proj_corners2D[j, 1]

        vertices_2D.append(vertices_proj_2d_colored)
        # triangles_2D.append(triangles)

    class_names = load_class_names(namesfile)
    plot_boxes(img, boxes, 'predictions.jpg', class_names, vertices_2D)





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
