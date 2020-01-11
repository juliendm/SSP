import sys
import time
from PIL import Image, ImageDraw
#from models.tiny_yolo import TinyYoloNet
from utils import *
from image import letterbox_image, correct_yolo_boxes
from darknet import Darknet
import pickle


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

    if imgfile == 'None':
        with open('../SSP/yolov3/cfg/valid_pku_baidu.txt', 'r') as subm_file:
           imgfiles = subm_file.readlines()
    else:
        imgfiles = [imgfile]
    print('Loading Image Files... Done')

    for index,imgfile in enumerate(imgfiles):
        if index%50==0: print(index)

        imgpath = imgfile.rstrip()[3:]
        maskpath = imgpath.replace('images', 'masks').replace('JPEGImages', 'masks')
        img_id = imgpath.split('/')[-1].split('.')[0]   

        imgpath_preprocessed = imgpath.replace('images', 'images_prepro')

        try:
            img = Image.open(imgpath_preprocessed).convert('RGB')
        except OSError:
            img = Image.open(imgpath).convert('RGB').crop((0,1497,3384,2710))
            try: #  it avoids the unnecessary call to os.path.exists()
                mask = Image.open(maskpath).convert('L').crop((0,1497,3384,2710))
                white_img = Image.new('RGB', img.size, (255, 255, 255))
                img = Image.composite(white_img, img, mask)
            except OSError:
                pass
            # img.save(imgpath_preprocessed)

        sized = letterbox_image(img, m.width, m.height)

        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        correct_yolo_boxes(boxes, img.width, img.height, m.width, m.height)

        finish = time.time()
        print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

        boxes = np.array(boxes)
        mask = np.ones(boxes.shape[0], dtype=bool)
        mask = np.logical_and(mask, boxes[:,6] == 2)

        with open('../baidu_data/predictions_yolov3/%s.pkl' % img_id,'wb') as pkl_file:
            pickle.dump(boxes[mask], pkl_file)
        class_names = load_class_names(namesfile)
        plot_boxes(img, boxes[mask], '../baidu_data/predictions_yolov3/%s.jpg' % img_id, class_names)

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
