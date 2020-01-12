import math
import numpy as np
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import bbox_iou, multi_bbox_ious, convert2cpu

class YoloLayer(nn.Module):
    def __init__(self, anchor_mask=[], num_classes=0, anchors=[1.0], num_anchors=1, use_cuda=None):
        super(YoloLayer, self).__init__()
        use_cuda = torch.cuda.is_available() and (True if use_cuda is None else use_cuda)
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.anchor_mask = anchor_mask
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors)//num_anchors
        self.rescore = 1
        self.ignore_thresh = 0.5
        self.truth_thresh = 1.
        self.nth_layer = 0
        self.seen = 0
        self.net_width = 0
        self.net_height = 0

    def get_mask_boxes(self, output):
        masked_anchors = []
        for m in self.anchor_mask:
            masked_anchors += self.anchors[m*self.anchor_step:(m+1)*self.anchor_step]

        masked_anchors = torch.FloatTensor(masked_anchors).to(self.device)
        num_anchors = torch.IntTensor([len(self.anchor_mask)]).to(self.device)
        return {'x':output, 'a':masked_anchors, 'n':num_anchors}

    def build_targets(self, pred_boxes, target, anchors, nA, nH, nW):

        num_keypoints = 1
        num_labels = 12

        nB = target.size(0)
        anchor_step = anchors.size(1) # anchors[nA][anchor_step]
        noobj_mask = torch.ones (nB, nA, nH, nW)
        obj_mask   = torch.zeros(nB, nA, nH, nW)
        coord_mask = torch.zeros(nB, nA, nH, nW)
        tcoord     = torch.zeros(num_labels-1, nB, nA, nH, nW)
        tconf      = torch.zeros(nB, nA, nH, nW)
        tcls       = torch.zeros(nB, nA, nH, nW, self.num_classes)

        nAnchors = nA*nH*nW
        nPixels  = nH*nW
        nGT = 0
        nRecall = 0
        nRecall75 = 0

        # it works faster on CPU than on GPU.
        anchors = anchors.to("cpu")

        for b in range(nB):
            cur_pred_boxes = pred_boxes[b*nAnchors:(b+1)*nAnchors,[0,1,2*num_keypoints+7,2*num_keypoints+7+1]].t()  # Filter!!!
            cur_ious = torch.zeros(nAnchors)
            tbox = target[b].view(-1,num_labels).to("cpu")

            for t in range(50):
                if tbox[t][1] == 0:
                    break
                gx, gy = tbox[t][1] * nW, tbox[t][2] * nH
                gw, gh = tbox[t][-2] * self.net_width, tbox[t][-1] * self.net_height
                cur_gt_boxes = torch.FloatTensor([gx, gy, gw, gh]).repeat(nAnchors,1).t()
                cur_ious = torch.max(cur_ious, multi_bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
            ignore_ix = (cur_ious>self.ignore_thresh).view(nA,nH,nW)
            noobj_mask[b][ignore_ix] = 0

            for t in range(50):
                if tbox[t][1] == 0:
                    break
                nGT += 1

                gx, gy = tbox[t][1] * nW, tbox[t][2] * nH
                gw, gh = tbox[t][2*num_keypoints+7+1] * self.net_width, tbox[t][2*num_keypoints+7+2] * self.net_height
                gw, gh = gw.float(), gh.float()
                gi, gj = int(gx), int(gy)

                tmp_gt_boxes = torch.FloatTensor([0, 0, gw, gh]).repeat(nA,1).t()
                anchor_boxes = torch.cat((torch.zeros(nA, anchor_step), anchors),1).t()
                _, best_n = torch.max(multi_bbox_ious(anchor_boxes, tmp_gt_boxes, x1y1x2y2=False), 0)

                gt_box = torch.FloatTensor([gx, gy, gw, gh])
                pred_box = pred_boxes[b*nAnchors+best_n*nPixels+gj*nW+gi,[0,1,2*num_keypoints+7,2*num_keypoints+7+1]] # Filter!!!
                iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)

                obj_mask  [b][best_n][gj][gi] = 1
                noobj_mask[b][best_n][gj][gi] = 0
                coord_mask[b][best_n][gj][gi] = 2. - tbox[t][2*num_keypoints+7+1]*tbox[t][2*num_keypoints+7+2]

                for i in range(num_keypoints):
                    tcoord[2*i][b][best_n][gj][gi]   = tbox[t][2*i+1] * nW - gi
                    tcoord[2*i+1][b][best_n][gj][gi] = tbox[t][2*i+2] * nH - gj

                for i in range(7):
                    tcoord[2*num_keypoints+i][b][best_n][gj][gi]   = tbox[t][2*num_keypoints+i+1]

                tcoord[2*num_keypoints+7][b][best_n][gj][gi] = math.log(gw/anchors[best_n][0])
                tcoord[2*num_keypoints+7+1][b][best_n][gj][gi] = math.log(gh/anchors[best_n][1])
                tcls      [b][best_n][gj][gi][int(tbox[t][0])] = 1
                tconf     [b][best_n][gj][gi] = iou if self.rescore else 1.

                if iou > 0.5:
                    nRecall += 1
                    if iou > 0.75:
                        nRecall75 += 1

        return nGT, nRecall, nRecall75, obj_mask, noobj_mask, coord_mask, tcoord, tconf, tcls

    def forward(self, output, target):

        num_keypoints = 1
        num_labels = 12

        #output : BxAs*(4+1+num_classes)*H*W
        mask_tuple = self.get_mask_boxes(output)
        t0 = time.time()
        nB = output.data.size(0)    # batch size
        nA = mask_tuple['n'].item() # num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)
        anchor_step = mask_tuple['a'].size(0)//nA
        anchors = mask_tuple['a'].view(nA, anchor_step).to(self.device)
        cls_anchor_dim = nB*nA*nH*nW

        output  = output.view(nB, nA, (num_labels+nC), nH, nW)
        cls_grid = torch.linspace(num_labels,num_labels+nC-1,nC).long().to(self.device)
        ix = torch.LongTensor(range(0,num_labels)).to(self.device)
        pred_boxes = torch.FloatTensor(num_labels-1, cls_anchor_dim).to(self.device)

        coord = output.index_select(2, ix[0:num_labels-1]).view(nB*nA, -1, nH*nW).transpose(0,1).contiguous().view(-1,cls_anchor_dim)  # x1, y1, x2, y2, ...
        coord[0:2] = coord[0:2].sigmoid()
        conf = output.index_select(2, ix[num_labels-1]).view(cls_anchor_dim).sigmoid()

        cls  = output.index_select(2, cls_grid)
        cls  = cls.view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(cls_anchor_dim, nC).to(self.device)

        t1 = time.time()
        grid_x = torch.linspace(0, nW-1, nW).repeat(nB*nA, nH, 1).view(cls_anchor_dim).to(self.device)
        grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(cls_anchor_dim).to(self.device)
        anchor_w = anchors.index_select(1, ix[0]).repeat(nB, nH*nW).view(cls_anchor_dim)
        anchor_h = anchors.index_select(1, ix[1]).repeat(nB, nH*nW).view(cls_anchor_dim)

        for i in range(num_keypoints):
            pred_boxes[2*i]   = coord[2*i]   + grid_x
            pred_boxes[2*i+1] = coord[2*i+1] + grid_y
        for i in range(7):
            pred_boxes[2*num_keypoints+i]   = coord[2*num_keypoints+i]
        pred_boxes[2*num_keypoints+7]   = coord[2*num_keypoints+7].exp() * anchor_w
        pred_boxes[2*num_keypoints+7+1] = coord[2*num_keypoints+7+1].exp() * anchor_h

        # for build_targets. it works faster on CPU than on GPU
        pred_boxes = convert2cpu(pred_boxes.transpose(0,1).contiguous().view(-1,num_labels-1)).detach()

        t2 = time.time()
        nGT, nRecall, nRecall75, obj_mask, noobj_mask, coord_mask, tcoord, tconf, tcls = \
            self.build_targets(pred_boxes, target.detach(), anchors.detach(), nA, nH, nW)

        conf_mask = (obj_mask + noobj_mask).view(cls_anchor_dim).to(self.device)
        obj_mask  = (obj_mask==1).view(cls_anchor_dim)

        nProposals = int((conf > 0.25).sum())

        coord = coord[:,obj_mask]
        tcoord = tcoord.view(num_labels-1, cls_anchor_dim)[:,obj_mask].to(self.device)        

        tconf = tconf.view(cls_anchor_dim).to(self.device)        

        cls = cls[obj_mask,:].to(self.device)
        tcls = tcls.view(cls_anchor_dim, nC)[obj_mask,:].to(self.device)

        t3 = time.time()
        loss_coord  = nn.BCELoss(reduction='sum')(coord[0:2], tcoord[0:2])/nB + \
                      nn.MSELoss(reduction='sum')(coord[2*num_keypoints+7:2*num_keypoints+7+2], tcoord[2*num_keypoints+7:2*num_keypoints+7+2])/nB
        
        trans_pred = coord[2*num_keypoints:2*num_keypoints+3]
        label_trans = tcoord[2*num_keypoints:2*num_keypoints+3]
        #loss_trans = nn.MSELoss(reduction='sum')(trans_pred, label_trans)/nB
        loss_trans = huber_loss(trans_pred, label_trans, self.device)/nB

        rot_pred = coord[2*num_keypoints+3:2*num_keypoints+7]
        label_rot = tcoord[2*num_keypoints+3:2*num_keypoints+7]
        rot_pred = F.normalize(rot_pred, p=2, dim=0)
        #loss_rot = nn.MSELoss(reduction='sum')(rot_pred, label_rot)/nB
        loss_rot = torch.abs(rot_pred - tcoord[2*num_keypoints+3:2*num_keypoints+7])
        loss_rot = loss_rot.view(-1).sum(0) / nB

        loss_conf   = nn.BCELoss(reduction='sum')(conf*conf_mask, tconf*conf_mask)/nB
        loss_cls    = nn.BCEWithLogitsLoss(reduction='sum')(cls, tcls)/nB

        loss = loss_coord + loss_trans + loss_rot + loss_conf + loss_cls

        t4 = time.time()
        if False:
            print('-'*30)
            print('        activation : %f' % (t1 - t0))
            print(' create pred_boxes : %f' % (t2 - t1))
            print('     build targets : %f' % (t3 - t2))
            print('       create loss : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))
            
        #if (self.seen-self.seen//100*100) < nB:
        print('%d: Layer(%03d) nGT %3d, nRC %3d, nRC75 %3d, nPP %3d, loss: coord %6.3f, trans %6.3f, rot %6.3f, conf %6.3f, class %6.3f, total %7.3f' 
                % (self.seen, self.nth_layer, nGT, nRecall, nRecall75, nProposals, loss_coord, loss_trans, loss_rot, loss_conf, loss_cls, loss))
        if math.isnan(loss.item()):
            print(coord, conf, tconf)
            sys.exit(0)
        return loss


def huber_loss(bbox_pred, bbox_targets, device, beta=2.8/100.0):
    """
    SmoothL1(x) = 0.5 * x^2 / beta      if |x| < beta
                  |x| - 0.5 * beta      otherwise.
    https://en.wikipedia.org/wiki/Huber_loss
    """
    box_diff = bbox_pred - bbox_targets

    dis_trans = np.linalg.norm(box_diff.data.cpu().numpy(), axis=1)
    # we also add a metric for dist<2.8 metres.
    inbox_idx = dis_trans <= 2.8/100.0
    outbox_idx = dis_trans > 2.8/100.0

    bbox_inside_weights = torch.autograd.Variable(torch.from_numpy(inbox_idx.astype('float32'))).to(device)
    bbox_outside_weights = torch.autograd.Variable(torch.from_numpy(outbox_idx.astype('float32'))).to(device)

    in_box_pow_diff = 0.5 * torch.pow(box_diff, 2) / beta
    in_box_loss = in_box_pow_diff.sum(dim=1) * bbox_inside_weights

    out_box_abs_diff = torch.abs(box_diff)
    out_box_loss = (out_box_abs_diff.sum(dim=1) - beta / 2) * bbox_outside_weights

    loss_box = in_box_loss + out_box_loss
    loss_box = loss_box.view(-1).sum(0)

    return loss_box
