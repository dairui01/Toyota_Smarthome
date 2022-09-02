import numpy as np
import cv2

image_path="/u/srvpal/data/people/rdai/smarthome/P11T13C03/00000001.jpg"
skeleton=np.load("P11T13C03_LCRNet2d.npz")['kpts']
img = cv2.imread(image_path)

red = [0,0,255]
blue = [255,0,0]

for i in range(1,17):
    x,y=skeleton[0][i]
    # img[int(x),int(y)]=red
    img = cv2.circle(img, (x,y),radius=4, color=(0, 0, 255), thickness=-1)


skeleton_set=[[6,5],[6,8],[8,10],[5,7],[7,9],[12,11],[12,14],[14,16],[11,13],[13,15]]
for line in skeleton_set:
    x1,y1=skeleton[0][line[0]]
    x2,y2=skeleton[0][line[1]]
    img = cv2.line(img, (x1,y1), (x2,y2), blue, thickness=2)

mid_point_x=int(0.5*(skeleton[0][11][0]+skeleton[0][12][0]))
mid_point_y=int(0.5*(skeleton[0][11][1]+skeleton[0][12][1]))

img = cv2.line(img, (int(skeleton[0][4][0]),int(skeleton[0][4][1])), (mid_point_x,mid_point_y), blue, thickness=2)

cv2.imwrite('./test.jpg', img)

