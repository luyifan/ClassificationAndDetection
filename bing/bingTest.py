from bing import Bing
from bing import Boxes
b = Bing(2,8,2);
b.loadTrainModel("/home/deepnet/Desktop/VOC2007/Results/ObjNessB2W8MAXBGR")
boxes=b.getBoxesOfOneImage("/home/deepnet/Desktop/VOC2007/JPEGImages/000220.jpg",100,"/home/deepnet/Desktop/1.txt")
print [ s for s in boxes.ymins() ]
print [ s for s in boxes.ymaxs() ]
