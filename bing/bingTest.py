from bing import Bing
from bing import Boxes
import numpy as np
import pandas as pd 
import time
import cv2 
from sklearn import cluster
b = Bing(2,8,2);
imageFilename="/home/deepnet/Desktop/VOC2007/JPEGImages/000043.jpg"
cluster_num=20
b.loadTrainModel("/home/deepnet/Desktop/VOC2007/Results/ObjNessB2W8MAXBGR")
boxes=b.getBoxesOfOneImage(imageFilename,130)
values=[ s for s in boxes.values() ]
print values
ymins=[ s for s in boxes.ymins() ]
ymaxs=[ s for s in boxes.ymaxs() ]
xmins=[ s for s in boxes.xmins() ]
xmaxs=[ s for s in boxes.xmaxs() ]
bing_windows=pd.DataFrame({0:ymins,1:xmins,2:ymaxs,3:xmaxs})
windows=bing_windows.values
windows_size=bing_windows.shape[0]
starttime=time.time()
y1=windows[:,0]
x1=windows[:,1]
y2=windows[:,2]
x2=windows[:,3]
w=x2-x1
h=y2-y1
area=(w*h).astype(float)
distances=np.zeros((windows_size,windows_size))
for i in range(windows_size):
    xx1 = np.maximum(x1[i],x1)
    yy1 = np.maximum(y1[i],y1)
    xx2 = np.minimum(x2[i],x2)
    yy2 = np.minimum(y2[i],y2)
    w = np.maximum(0.,xx2-xx1)
    h = np.maximum(0.,yy2-yy1)
    wh = w*h
    o = wh/(area[i]+area-wh)
    distances[i]=o
endtime=time.time()
print "{:.3f}".format(endtime-starttime)
starttime=time.time()
spectral=cluster.SpectralClustering(n_clusters=10,affinity='precomputed')
spectral.fit(distances)
#print spectral.labels_
endtime=time.time()
color=np.random.randint(0,255,(20,3))

img=cv2.imread(imageFilename)
for i in range(windows_size):
    label=spectral.labels_[i]
    #print label
    #print color[label].tolist()
    cv2.rectangle(img,(x1[i],y1[i]),(x2[i],y2[i]),color[label].tolist(),1)

cv2.imwrite("test.jpg",img)

print "{:.3f}".format(endtime-starttime)
