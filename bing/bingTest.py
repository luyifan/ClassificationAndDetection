from bing import Bing
from bing import Boxes
import numpy as np
import pandas as pd 
import time
import cv2 
import itertools
from sklearn import cluster
b = Bing(2,8,2);
imageFilename="/home/deepnet/Desktop/VOC2007/JPEGImages/000019.jpg"
cluster_num=3
top_k=5
max_ratio=4
min_size=100
b.loadTrainModel("/home/deepnet/Desktop/VOC2007/Results/ObjNessB2W8MAXBGR")
boxes=b.getBoxesOfOneImage(imageFilename,130)
values=[ s for s in boxes.values() ]
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
spectral=cluster.SpectralClustering(n_clusters=cluster_num,affinity='precomputed')
spectral.fit(distances)

endtime=time.time()
print "{:.3f}".format(endtime-starttime)
#print spectral.labels_
#calculate the top K in each cluster 
starttime=time.time()
index_dictionary={}
w=x2-x1
h=y2-y1
for i in range(windows_size):
    if(area[i]<min_size):
        continue
    #print "{:.3f} , {:.3f} , {:.3f}".format(w[i],h[i],w[i]*1.0/h[i])
    if(w[i]*1.0/h[i]>max_ratio or h[i]*1.0/w[i]>max_ratio):
        continue
    label=spectral.labels_[i]
    if not label in index_dictionary:
        index_dictionary[label]=[]
    if len(index_dictionary[label])>top_k:
        continue
    index_dictionary[label].append(i)

index_list=[]
print index_dictionary
for key in index_dictionary:
    index_list=itertools.chain(index_list,index_dictionary[key])
index_list=[ s for s in index_list]
x1=x1[index_list]
y1=y1[index_list]
x2=x2[index_list]
y2=y2[index_list]
label=spectral.labels_[index_list]
windows_size=len(index_list)
endtime=time.time()
color=np.random.randint(0,255,(cluster_num,3))
img=cv2.imread(imageFilename)
for i in range(windows_size):
    #print label
    #if not (label[i] == 0):
    #    continue
    #print color[label].tolist()
    cv2.rectangle(img,(x1[i],y1[i]),(x2[i],y2[i]),color[label[i]].tolist(),3)

cv2.imwrite("test.jpg",img)
endtime=time.time()
print "{:.3f}".format(endtime-starttime)
