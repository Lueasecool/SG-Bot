
import numpy as np
import torch


def validate_constrains(triples, pred_boxes, pred_angles, keep, vocab, accuracy, strict=True, overlap_threshold=0.3):

    param6 = pred_boxes.shape[1] == 6
    layout_boxes = pred_boxes

    for [s, p, o] in triples:
        if keep is None:
            box_s = layout_boxes[s.item()].cpu().detach().numpy()
            box_o = layout_boxes[o.item()].cpu().detach().numpy()
        else:
            if keep[s.item()] == 1 and keep[o.item()] == 1: # if both are unchanged we evaluate the normal constraints
                box_s = layout_boxes[s.item()].cpu().detach().numpy()
                box_o = layout_boxes[o.item()].cpu().detach().numpy()
            else:
                continue

        if vocab["pred_idx_to_name"][p.item()][:-1] == "left":
            # z
            if box_s[5] - box_o[5] > -0.05 or (strict and box3d_iou(box_s, box_o, param6=param6, with_translation=True)[0] > overlap_threshold):
                accuracy['left'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['left'].append(1)
                accuracy['total'].append(1)
        if vocab["pred_idx_to_name"][p.item()][:-1] == "right":
            if box_s[5] - box_o[5] < 0.05 or (strict and box3d_iou(box_s, box_o, param6=param6, with_translation=True)[0] > overlap_threshold):
                accuracy['right'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['right'].append(1)
                accuracy['total'].append(1)
        if vocab["pred_idx_to_name"][p.item()][:-1] == "front":
            if box_s[3] - box_o[3] < -0.05 or (strict and box3d_iou(box_s, box_o, param6=param6, with_translation=True)[0] > overlap_threshold):
                accuracy['front'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['front'].append(1)
                accuracy['total'].append(1)
        if vocab["pred_idx_to_name"][p.item()][:-1] == "behind":
            if box_s[3] - box_o[3] > 0.05 or (strict and box3d_iou(box_s, box_o, param6=param6, with_translation=True)[0] > overlap_threshold):
                accuracy['behind'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['behind'].append(1)
                accuracy['total'].append(1)
        # bigger than
        if vocab["pred_idx_to_name"][p.item()][:-1] == "bigger than":
            sub_volume = box_s[0] * box_s[1] * box_s[2]
            obj_volume = box_o[0] * box_o[1] * box_o[2]
            if (sub_volume - obj_volume) / sub_volume < 0.15:
                accuracy['bigger'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['bigger'].append(1)
                accuracy['total'].append(1)
        # smaller than
        if vocab["pred_idx_to_name"][p.item()][:-1] == "smaller than":
            sub_volume = box_s[0] * box_s[1] * box_s[2]
            obj_volume = box_o[0] * box_o[1] * box_o[2]
            if (sub_volume - obj_volume) / sub_volume > -0.15:
                accuracy['smaller'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['smaller'].append(1)
                accuracy['total'].append(1)
        # higher than
        if vocab["pred_idx_to_name"][p.item()][:-1] == "taller than":
            absheight_s = box_s[4]+box_s[1]
            absheight_o = box_o[4]+box_o[1]
            if (absheight_s - absheight_o) / absheight_s < 0.1:
                accuracy['taller'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['taller'].append(1)
                accuracy['total'].append(1)
        # lower than
        if vocab["pred_idx_to_name"][p.item()][:-1] == "shorter than":
            absheight_s = box_s[4] + box_s[1]
            absheight_o = box_o[4] + box_o[1]
            if (absheight_s - absheight_o) / absheight_s > -0.1:
                accuracy['shorter'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['shorter'].append(1)
                accuracy['total'].append(1)

        # standing on
        if vocab["pred_idx_to_name"][p.item()][:-1] == "standing on":
            if np.abs(box_s[4] - box_o[4]) < 0.04:
                accuracy['standing on'].append(1)
                accuracy['total'].append(1)
            else:
                accuracy['standing on'].append(0)
                accuracy['total'].append(0)
        # close by
        if vocab["pred_idx_to_name"][p.item()][:-1] == "close by":
            corners_s = corners_from_box(box_s, param6, with_translation=True)
            corners_o = corners_from_box(box_o, param6, with_translation=True)
            c_dist1 = close_dis(corners_s, corners_o)
            if c_dist1 > 0.45:
                accuracy['close by'].append(0)
                accuracy['total'].append(0)
            else:
                accuracy['close by'].append(1)
                accuracy['total'].append(1)

        # symmetrical to
        if vocab["pred_idx_to_name"][p.item()][:-1] == "symmetrical to":
            sub_center_in_scene_flip_x = [-box_s[3], box_s[5]]
            sub_center_in_scene_flip_z = [box_s[3], -box_s[5]]
            sub_center_in_scene_flip_xz = [-box_s[3], -box_s[5]]
            obj_center_in_scene = [box_o[3], box_o[5]]
            if cal_l2_distance(sub_center_in_scene_flip_xz, obj_center_in_scene) < 0.45 or \
                cal_l2_distance(sub_center_in_scene_flip_x, obj_center_in_scene) < 0.45 or \
                cal_l2_distance(sub_center_in_scene_flip_z, obj_center_in_scene) < 0.45:
                accuracy['symmetrical to'].append(1)
                accuracy['total'].append(1)
            else:
                accuracy['symmetrical to'].append(0)
                accuracy['total'].append(0)

    return accuracy




def corners_from_box(box, param6=True, with_translation=False):
    # box given as: [l, h, w, px, py, pz, z]
    # l meansures z axis; h measures y axis; w measures x axis.
    # (px, py, pz) is the bottom center
    if param6:
        l, h, w, px, py, pz = box
    else:
        l, h, w, px, py, pz, _ = box

    (tx, ty, tz) = (px, py, pz) if with_translation else (0,0,0)

    x_corners = [w/2,w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2]
    y_corners = [h,h,h,h,0,0,0,0]
    z_corners = [l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2,l/2]
    corners_3d = np.dot(np.eye(3), np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + tx
    corners_3d[1,:] = corners_3d[1,:] + ty
    corners_3d[2,:] = corners_3d[2,:] + tz
    corners_3d = np.transpose(corners_3d)

    return corners_3d



def box3d_iou(box1, box2, param6=True, with_translation=False):
    ''' Compute 3D bounding box IoU.
    Input:
        corners1: numpy array (8,3), assume up direction is positive Y_h
        corners2: numpy array (8,3), assume up direction is positive Y_h
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU
    '''
    # corner points are in counter clockwise order
    corners1 = corners_from_box(box1, param6, with_translation)
    corners2 = corners_from_box(box2, param6, with_translation)

    rect1 = [(corners1[i,2], corners1[i,0]) for i in range(0,4)]
    rect2 = [(corners2[i,2], corners2[i,0]) for i in range(0,4)]

    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])

    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])

    inter_vol = inter_area * max(0.0, ymax-ymin)

    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)

    volmin = min(vol1, vol2)

    iou = inter_vol / volmin #(vol1 + vol2 - inter_vol)

    return iou, iou_2d