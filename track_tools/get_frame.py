import cv2
import os.path as osp
import os

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


type = 'test'
wh = (1422, 800)

seq_root = 'C:/Users/ippr00/Downloads/TransTrack/TransTrack/boat_dataset/' + type 
seqs = [s for s in os.listdir(seq_root)]

for seq in seqs:

    video = osp.join(seq_root, seq, '{}.mp4'.format(seq)) 
    dst_path = osp.join(seq_root, seq, 'img1')
    mkdirs(dst_path)

    videoCapture = cv2.VideoCapture()
    videoCapture.open(video)

    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)

    print(seq, 'fps:{}, frames:{}'.format(fps, frames))

    for i in range(int(frames)):
        print('seq: %s frame: %d\r' %(seq, i), end='\r')
        ret, frame = videoCapture.read()
        
        try:
            frame = cv2.resize(frame, wh, interpolation=cv2.INTER_AREA)
        except:
            print('image:{}'.format(i))
        cv2.imwrite(osp.join(dst_path, '{0:06d}.jpg'.format(i)), frame)

# video = 'C:/Users/ippr00/Downloads/TransTrack/TransTrack/boat_dataset/test/012_B/012_B.mp4'
# dst_path = 'C:/Users/ippr00/Downloads/TransTrack/TransTrack/boat_dataset/test/012_B/img1/'
# videoCapture = cv2.VideoCapture()
# videoCapture.open(video)

# video_capture = cv2.VideoCapture(video)
# frameToStart = 918
# video_capture.set(cv2.CAP_PROP_POS_FRAMES, frameToStart)
# ret,frame = video_capture.retrieve() # 此时返回的frame便是第25帧图像
# cv2.imshow('img',frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# frame = cv2.resize(frame, (1422,800), interpolation=cv2.INTER_AREA)
# print('resize sucess')
# cv2.imwrite(osp.join(dst_path, '{0:06d}.jpg'.format(frameToStart)) ,frame)

