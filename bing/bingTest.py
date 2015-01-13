from bing import Bing
from bing import Boxes
import numpy as np
import pandas as pd 
b = Bing(2,8,2);
b.loadTrainModel("/home/deepnet/Desktop/VOC2007/Results/ObjNessB2W8MAXBGR")
boxes=b.getBoxesOfOneImage("/home/deepnet/Desktop/VOC2007/JPEGImages/000220.jpg",130)
ymins=[ s for s in boxes.ymins() ]
ymaxs=[ s for s in boxes.ymaxs() ]
xmins=[ s for s in boxes.xmins() ]
xmaxs=[ s for s in boxes.xmaxs() ]
bing_windows=pd.DataFrame({0:ymins,1:xmins,2:ymaxs,3:xmaxs})
print bing_windows.head(1)
