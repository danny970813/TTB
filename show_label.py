import cv2
import numpy as np
import os
import os.path as osp
from track_tools.colormap import colormap

def load_mot(detections):
    """
    Loads detections stored in a mot-challenge like formatted CSV or numpy array (fieldNames = ['frame', 'id', 'x', 'y',
    'w', 'h', 'score']).
    Args:
        detections (str, numpy.ndarray): path to csv file containing the detections or numpy array containing them.
    Returns:
        list: list containing the detections for each frame.
    """
    
    data = []
    if type(detections) is str:
        raw = np.genfromtxt(detections, delimiter=',', dtype=np.float32)
    else:
        # assume it is an array
        assert isinstance(detections, np.ndarray), "only numpy arrays or *.csv paths are supported as detections."
        raw = detections.astype(np.float32)

    end_frame = int(np.max(raw[:, 0]))
    for i in range(1, end_frame+1):
        idx = raw[:, 0] == i
        id, bbox = raw[idx, 1], raw[idx, 2:6]
        bbox[:, 2:4] += bbox[:, 0:2]  # x1, y1, w, h -> x1, y1, x2, y2
        bbox -= 1  # correct 1,1 matlab offset
        scores = raw[idx, 6]
        dets = []
        for idn, bb, s in zip(id, bbox, scores):
            dets.append({'id':int(idn), 'bbox': (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])), 'score': s})
        data.append(dets)

    return data

def mkdir_if_missing(dir):
    os.makedirs(dir, exist_ok=True)

def images_to_video(path, video_name, start_frame):
    fps = 30  # 帧率
    dirs = os.listdir(path)
    num_frames = len(dirs)
    img_array = []
    img_width = 1422
    img_height = 800
    for i in range(1 + start_frame, start_frame + num_frames + 1):
        print('img: %s\r' %str(i).rjust(5, '0'), end='')
        filename = path + "/"+ str(i).rjust(5, '0')+".jpg"
        img = cv2.imread(filename)
 
        if img is None:
            print(filename + " is non-existent!")
            continue
        img_array.append(img)
 
    out = cv2.VideoWriter(path[:11] + '/' + video_name + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps,(img_width,img_height))
 
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def main():

    SHOW = False
    combine = True
    
    TYPE = 'train'
    
    seq_root = 'boat_dataset/'+ TYPE 
    seqs = [s for s in os.listdir(seq_root)]
    
    for seq in seqs:
        # video_name = '001_A'
        print('{} processing...'.format(seq), end='\n')
        frame_path = osp.join(seq_root, seq, 'img1') 
        det_dir = osp.join('demo_output/demo_images/', seq)
        GT = []

    # video_name = '003_B'
    # frame_path = 'boat_dataset/test/' + video_name + '/img1' 
    # det_dir = 'demo_output/demo_images/' + video_name
    # print(det_dir[:11])
        if not os.path.exists(frame_path):
            print('vedio is not exist')

        if det_dir:
            mkdir_if_missing(det_dir)
        
        n_frame = len(os.listdir(frame_path))
        # vid = cv2.VideoCapture(video_path)
        if TYPE == 'train':
            dets = load_mot('output/val/tracks/' + seq + '.txt')
            start_frame = round(n_frame/2)
        else:
            dets = load_mot('output/test/tracks/' + seq + '.txt')
            start_frame = 1
        color_list = colormap()
        
        

        for frame_num, detections_frame in enumerate(dets, start = start_frame):
            frame =cv2.imread(frame_path + '/' + str(frame_num).zfill(6) + '.jpg')
            for a in range(len(detections_frame)):
                id, bbox = detections_frame[a]["id"], detections_frame[a]["bbox"]
                if SHOW == True:
                    print(frame_num, 'id', id, "bbox:x1,y1,x2,y2", bbox)
                # cv2.rectangle(frame,bbox[:2], bbox[2:], (255,0,0), 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color_list[id%79].tolist(), thickness=2)
                cv2.putText(frame,"{}".format(id), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_list[id%79].tolist(), 2)
                if SHOW == True:
                    cv2.imshow("frame", frame)

            if det_dir is not None:
                cv2.imwrite(os.path.join(det_dir, '{:05d}.jpg'.format(frame_num)), frame)

            if SHOW == True:
                key = cv2.waitKey(100)& 0xFF
                if key == ord(' '):
                    cv2.waitKey(0)
                if key == ord('q'):
                    break
            if SHOW == False:
                print('frame:{}'.format(frame_num), sep=' ', end='\r')

        
        if combine == True:
            images_to_video(det_dir, seq, start_frame)
        

if __name__ == '__main__':
    main()
    




 
 
