
import os.path as osp
import os
import numpy as np
import shutil

# def xyxy2xywh(x):
#     # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
#     # y = torch.zeros_like(x) if isinstance(x,
#     #                                       torch.Tensor) else np.zeros_like(x)
#     y = [0, 0, 0, 0]

#     y[0] = (x[0] + x[2]) / 2
#     y[1] = (x[1] + x[3]) / 2
#     y[2] = x[2] - x[0]
#     y[3] = x[3] - x[1]
#     return y



def cal_scale(wh, n_wh):
    w_scale = n_wh[0]/wh[0]
    h_scale = n_wh[1]/wh[1]
    return (w_scale, h_scale)

def cal_center(bbox):
    x = bbox[0] + bbox[2]/2
    y = bbox[1] + bbox[3]/2

    return (x,y)

def resize(bbox, scale):
    center = cal_center(bbox)
    
    new_center = (center[0]*scale[0], center[1]*scale[1])
    new_w = bbox[2]*scale[0]
    new_h = bbox[3]*scale[1]
    new_x1 = new_center[0] - new_w/2
    new_y1 = new_center[1] - new_h/2

    return (new_x1, new_y1, new_w, new_h)
 
def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

wh = (1920,1080)
n_wh =(1422, 800)
resize_mode = True

type = 'train'

seq_root = 'C:/Users/ippr00/Downloads/TransTrack/TransTrack/boat_dataset/' + type 
seqs = [s for s in os.listdir(seq_root)]

for seq in seqs:
    # video_name = '001_A'
    print('{} processing...'.format(seq), end='\n')
    gt_path = osp.join(seq_root, seq, 'gt') 
    dst_path = osp.join(gt_path, 'MOT_gt.txt')
    GT = []

    if os.path.exists(dst_path):
            with open(dst_path, 'w') as fw:
                fw.write('')
 
    with open(osp.join(gt_path, 'gt.txt'), 'r') as f:
        lines = f.readlines()

        for line in lines:
            # print('seq: %s frame: %d\r' %(seq, fid), end='')
            line = line.split(',')
            frame = int(line[0])
            classification = line[1]
            obj_id = int(line[2])
            bbox = [line[3], line[4], line[5], line[6]]
            bbox = [int(i) for i in bbox]
            if (resize_mode == True):
                scale = cal_scale(wh, n_wh)
                new_bbox = resize(bbox, scale)
                # print('bbox:', bbox, 'new:', new_bbox)
            # wh_bbox = xyxy2xywh(bbox)

            obj = (frame, obj_id, new_bbox)
            GT.append(obj)

    GT = sorted(GT, key=lambda x:x[1])

    for content in GT:
        if os.path.exists(dst_path):
            with open(dst_path, 'a') as fw:
                fw.write('{},{},{},{},{},{},1,1,1\n'.format(content[0], content[1], content[2][0], content[2][1], content[2][2], content[2][3]))
        else:
            with open(dst_path, 'w') as fw:
                fw.write('{},{},{},{},{},{},1,1,1\n'.format(content[0], content[1], content[2][0], content[2][1], content[2][2], content[2][3]))
        