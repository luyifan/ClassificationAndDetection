from bing import Bing
b = Bing(2,8,2);
b.loadTrainModel("/home/deepnet/Desktop/VOC2007/Results/ObjNessB2W8MAXBGR")
b.getBoxesOfOneImage("/home/deepnet/Desktop/VOC2007/JPEGImages/000220.jpg",100,"/home/deepnet/Desktop/1.txt")
