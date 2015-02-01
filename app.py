import os
import time
import cPickle
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd
from PIL import Image as PILImage
import cStringIO as StringIO
import urllib
import caffe
import exifutil
import skimage.io
import cv2
from sklearn import cluster
import itertools
from bing import Bing
from bing import Boxes

PROJECT_DIRNAME = os.path.abspath(os.path.dirname(__file__))
IMAGE_PREFIX = "/static/temp/"
UPLOAD_FOLDER = PROJECT_DIRNAME + IMAGE_PREFIX

ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])
COORD_COLS = ['ymin', 'xmin', 'ymax', 'xmax']
RECTANGLE_COLOR = (0,0,255)
TEXT_COLOR = (255,0,0)
# Obtain the flask app object
app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False)

@app.route('/detection')
def detection():
    return flask.render_template('detection.html',has_result=False)

@app.route('/classification')
def classification():
    return flask.render_template('classification.html',has_result=False)

@app.route('/classify_url', methods=['GET'])
def classify_url():
    imageurl = flask.request.args.get('imageurl', '')
    try:
        string_buffer = StringIO.StringIO(
            urllib.urlopen(imageurl).read())
        image = caffe.io.load_image(string_buffer)

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
            'classification.html', has_result=True,
            result=(False, 'Cannot open image from URL.')
        )

    logging.info('Image: %s', imageurl)
    filename_ = str(datetime.datetime.now()).replace(' ','_') + \
            imageurl.split('/')[-1]
    filename = os.path.join(UPLOAD_FOLDER,filename_)
    skimage.io.imsave(filename,image)
    logging.info('Saving to %s.',filename)
    result = app.clf.classify_image(image)
    return flask.render_template(
        'classification.html', has_result=True, result=result, 
        imagesrc=embed_image_html(image)
        #imagesrc=IMAGE_PREFIX+filename_)
        )

@app.route('/detect_url',methods=['GET'])
def detect_url():
    imageurl = flask.request.args.get('imageurl','')
    try:
        string_buffer = StringIO.StringIO(
                urllib.urlopen(imageurl).read())
        image = caffe.io.load_image(string_buffer)
    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue 
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
                'detection.html', has_result=True,
                result=(False,'Cannot open image from URL.')
        )
    
    logging.info('Image: %s',imageurl)
    filename_ = str(datetime.datetime.now()).replace(' ','_') + \
            imageurl.split('/')[-1]
    filename = os.path.join(UPLOAD_FOLDER,filename_)
    skimage.io.imsave(filename,image)
    logging.info('Saving to %s.',filename)
    result = app.det.detect_image(str(filename))
    return flask.render_template(
            'detection.html' , has_result=True, result=result ,
            imagesrc=embed_image_html(image)
            #imagesrc=IMAGE_PREFIX+filename_)
            )

@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER,filename_)
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)
        image = exifutil.open_oriented_im(filename)
        #logging.info(image)
    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'classification.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )

    result = app.clf.classify_image(image)
    return flask.render_template(
        'classification.html', has_result=True, result=result,
        #imagesrc=IMAGE_PREFIX+filename_
        imagesrc=embed_image_html(image)
    )

@app.route('/detect_upload', methods=['POST'])
def detect_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ','_') + \
                werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER,filename_)
        imagefile.save(filename)
        logging.info('Saving to %s.',filename)
        #image = exifutil.open_oriented_im(filename)
        #logging.info(image)
    except Exception as err:
        logging.info('Uploaded image open error; %s' , err)
        return flask.render_template(
                'detection.html' , has_result=True,
                result=(False,'Cannot open uploded image.')
        )
    result = app.det.detect_image(str(filename))
    #print result
    image = exifutil.open_oriented_im(str(filename))
    return flask.render_template(
            'detection.html' , has_result=True,result=result,
            #imagesrc=IMAGE_PREFIX+filename_
            imagesrc=embed_image_html(image)
    )
@app.route('/detect_local',methods=['GET'])
def detect_local():
    filename_=flask.request.args.get('imagefilename','')
    if not allowed_file(filename_):
        logging.info('Detect Local Image Error, not a image file')
        return flask.render_template('detection.html',has_result=True,
                result=(False,'Detect error image')
                )
    imagefilename=PROJECT_DIRNAME+"/static/"+filename_
    if not os.path.isfile(imagefilename):
        logging.info('Detect Local Image Error,%s',imagefilename)
        return flask.render_template('detection.html',has_result=True,
                result=(False,'Detect image path error')
                )
    result = app.det.detect_image(str(imagefilename))
    image = exifutil.open_oriented_im(str(imagefilename))
    return flask.render_template(
           'detection.html' , has_result=True,result=result,
           #imagesrc="/static/"+filename_
           imagesrc=embed_image_html(image)
    )

@app.route('/classify_local',methods=['GET'])
def classify_local():
    filename_=flask.request.args.get('imagefilename','')
    if not allowed_file(filename_):
        logging.info('Classify Local Image Error, not a image file')
        return flask.render_template('classification.html',has_result=True,
                result=(False,'Classify error image')
        )
    imagefilename=PROJECT_DIRNAME+"/static/"+filename_
    if not os.path.isfile(imagefilename):
        logging.info('Classify Local Image Error,%s',imagefilename)
        return flask.render_template('classification.html',has_result=True,
                result=(False,'Classify image path error')
                )
    image=exifutil.open_oriented_im(imagefilename)
    result=app.clf.classify_image(image)
    return flask.render_template('classification.html',has_result=True,result=result,
            #imagesrc="/static/"+filename_
            imagesrc=embed_image_html(image)
            )


def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = PILImage.fromarray((255 * image).astype('uint8'))
    image_pil = image_pil.resize((256, 256))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )


class ImagenetClassifier(object):
    default_args = {
        'model_def_file': (
            '{}/models/bvlc_reference_caffenet/deploy.prototxt'.format(PROJECT_DIRNAME)),
        'pretrained_model_file': (
            '{}/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'.format(PROJECT_DIRNAME)),
        'mean_file': (
            '{}/models/ilsvrc12/ilsvrc_2012_mean.npy'.format(PROJECT_DIRNAME)),
        'class_labels_file': (
            '{}/models/ilsvrc12/synset_words.txt'.format(PROJECT_DIRNAME)),
        'bet_file': (
            '{}/models/ilsvrc12/imagenet.bet.pickle'.format(PROJECT_DIRNAME)),
    }
    for key, val in default_args.iteritems():
        if not os.path.exists(val):
            raise Exception(
                "File for {} is missing. Should be at: {}".format(key, val))
    default_args['image_dim'] = 227
    default_args['raw_scale'] = 255.
    default_args['gpu_mode'] = False

    def __init__(self, model_def_file, pretrained_model_file, mean_file,
                 raw_scale, class_labels_file, bet_file, image_dim, gpu_mode):
        logging.info('Loading net and associated files...')
        self.net = caffe.Classifier(
            model_def_file, pretrained_model_file,
            image_dims=(image_dim, image_dim), raw_scale=raw_scale,
            mean=np.load(mean_file), channel_swap=(2, 1, 0), gpu=gpu_mode
        )

        with open(class_labels_file) as f:
            labels_df = pd.DataFrame([
                {
                    'synset_id': l.strip().split(' ')[0],
                    'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
                }
                for l in f.readlines()
            ])
        self.labels = labels_df.sort('synset_id')['name'].values

        self.bet = cPickle.load(open(bet_file))
        # A bias to prefer children nodes in single-chain paths
        # I am setting the value to 0.1 as a quick, simple model.
        # We could use better psychological models here...
        self.bet['infogain'] -= np.array(self.bet['preferences']) * 0.1
    def classify_image(self, image):
        try:
            starttime = time.time()
            scores = self.net.predict([image], oversample=True).flatten()
	    endtime = time.time()

            indices = (-scores).argsort()[:5]
            predictions = self.labels[indices]

            # In addition to the prediction text, we will also produce
            # the length for the progress bar visualization.
            meta = [
                (p, '%.5f' % scores[i])
                for i, p in zip(indices, predictions)
            ]
            #logging.info('result: %s', str(meta))

            # Compute expected information gain
            expected_infogain = np.dot(
                self.bet['probmat'], scores[self.bet['idmapping']])
            expected_infogain *= self.bet['infogain']
            # sort the scores
            infogain_sort = expected_infogain.argsort()[::-1]
            bet_result = [(self.bet['words'][v], '%.5f' % expected_infogain[v])
                          for v in infogain_sort[:5]]
            logging.info('bet result: %s', str(bet_result))

            return (True, meta, bet_result, '%.3f' % (endtime - starttime))

        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')
class ImagenetDetection(object):
	default_args = {
			'model_def_file':(
				'{}/models/bvlc_reference_rcnn_ilsvrc13/deploy.prototxt'.format(PROJECT_DIRNAME)),
			'pretrained_model_file':(
				'{}/models/bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffemodel'.format(PROJECT_DIRNAME)),
			'mean_file': (
				'{}/models/ilsvrc12/ilsvrc_2012_mean.npy'.format(PROJECT_DIRNAME)),
			'class_labels_file':(
				'{}/models/ilsvrc12/det_synset_words.txt'.format(PROJECT_DIRNAME)),
        }
	for key , val in default_args.iteritems():
		if not os.path.exists(val):
			raise Exception("File for {} is missing. Should be at: {}".format(key, val))
	default_args['bing_model'] = '{}/models/bing/ObjNessB2W8MAXBGR'.format(PROJECT_DIRNAME)
        default_args['gpu_mode'] = False
        default_args['raw_scale'] = 255
        default_args['image_dim'] = 227
        default_args['channel_swap'] ='2,1,0'
        default_args['context_pad'] = 16
        default_args['cluster_num'] = 10
        default_args['top_k_in_cluster'] = 10
        default_args['max_ratio'] = 4
        default_args['min_size'] = 100
        def __init__(self,model_def_file, pretrained_model_file , mean_file , class_labels_file , bing_model , gpu_mode , raw_scale ,
                image_dim , channel_swap , context_pad , 
                cluster_num , top_k_in_cluster ,  max_ratio , min_size ,
                input_scale = None ):
		logging.info('Loading net and associated files...')
                mean , channel = None , None 
                if mean_file:
                    mean = np.load(mean_file)
                if channel_swap:
                    channel = [ int(s) for s in channel_swap.split(',') ]

                self.net = caffe.Detector(model_def_file,pretrained_model_file,
                        gpu=gpu_mode,mean=mean,input_scale=input_scale,raw_scale=raw_scale,channel_swap=channel,context_pad=context_pad)
                self.bing_search = Bing(2,8,2);
                self.bing_search.loadTrainModel(bing_model)
                if gpu_mode:
                    print 'GPU mode'
                else:
                    print 'CPU mode'
                with open(class_labels_file) as f:
                    labels_df = pd.DataFrame([
                        {
                            'synset_id' : l.strip().split(' ')[0],
                            'name' : ' '.join(l.strip().split(' ')[1:]).split(',')[0]
                        }
                        for l in f.readlines()
                        ])
                self.labels = labels_df.sort('synset_id')
                self.cluster_num = cluster_num
                self.top_k_in_cluster = top_k_in_cluster
                self.max_ratio = max_ratio
                self.min_size = min_size
                self.spectral = cluster.SpectralClustering(n_clusters=cluster_num,affinity='precomputed')
        def removeIOUandOverlap(self,i,index,x1,y1,x2,y2,area,iou,overlap):
            xx1 = np.maximum(x1[i],x1[index])
            yy1 = np.maximum(y1[i],y1[index])
            xx2 = np.minimum(x2[i],x2[index])
            yy2 = np.minimum(y2[i],y2[index])
            w = np.maximum(0.,xx2 - xx1)
            h = np.maximum(0.,yy2 - yy1)
            wh = w*h
            o = wh/(area[i]+area[index]-wh)
            oo = wh/np.minimum(area[i],area[index])
            first_match=np.nonzero(o<=iou)[0]
            second_match=np.nonzero(oo<=overlap)[0]
            index=index[np.intersect1d(second_match,first_match)]
            return index

        def nms_detections(self,dets,iou=0.1,overlap=0.8):
            x1 = dets[:,3]
            y1 = dets[:,2]
            x2 = dets[:,5]
            y2 = dets[:,4]
            ind = np.argsort(dets[:,0])
            dets_len = len(ind) 
            threshold = -0.2
            count = 0
            for i in ind:
                if dets[i,0]>=threshold:
                    ind_one = ind[:count]
                    ind_two = ind[count:]
                    break
                count+=1
                
            #pick=ind[:].tolist()[::-1]
            #return dets[pick,:]
            
            #if(dets_len <=1):
            #    pick=ind[:].tolist()[::-1]
            #    return dets[pick,:]
            #else:
            #    pick=ind[-1:].tolist()[::-1]
            w = x2 - x1
            h = y2 - y1 
            area = (w*h).astype(float)
            #ind=ind[:-1]
            pick=[]
            #pick=ind[:].tolist()[::-1]
            while len(ind_two)>0:
                i=ind_two[-1]
                pick.append(i)
                ind_two = ind_two[:-1]
                ind_two=self.removeIOUandOverlap(i,ind_two,x1,y1,x2,y2,area,iou*3,overlap)
                ind_one=self.removeIOUandOverlap(i,ind_one,x1,y1,x2,y2,area,iou,overlap)
            while len(ind_one)>0:
                i=ind_one[-1]
                pick.append(i)
                ind_one = ind_one[:-1]
                ind_one=self.removeIOUandOverlap(i,ind_one,x1,y1,x2,y2,area,iou,overlap)
            return dets[pick,:]
        
        def cluster_boxes(self,boxes): 
            ymins=np.array([ s for s in boxes.ymins() ]).astype(int)
            ymaxs=np.array([ s for s in boxes.ymaxs() ]).astype(int)
            xmins=np.array([ s for s in boxes.xmins() ]).astype(int)
            xmaxs=np.array([ s for s in boxes.xmaxs() ]).astype(int)
            #bing_windows=pd.DataFrame({0:ymins,1:xmins,2:ymaxs,3:xmaxs})
            #return bing_windows
            windows_size=len(xmins)
            width=xmaxs-xmins
            height=ymaxs-ymins
            area=(width*height).astype(float)
            distances=np.zeros((windows_size,windows_size))
            for i in range(windows_size):
                xx1 = np.maximum(xmins[i],xmins)
                yy1 = np.maximum(ymins[i],ymins)
                xx2 = np.minimum(xmaxs[i],xmaxs)
                yy2 = np.minimum(ymaxs[i],ymaxs)
                w = np.maximum(0.,xx2-xx1)
                h = np.maximum(0.,yy2-yy1)
                wh = w*h 
                distances[i]=wh/(area[i]+area-wh)
            starttimeInBoxes=time.time()
            self.spectral.fit(distances)
            endtimeInBoxes=time.time()
            logging.info("Cluster speend {:.3f}".format(endtimeInBoxes-starttimeInBoxes))
            starttimeInBoxes=time.time()
            index_dictionary={}
            for i in range(self.cluster_num):
                index_dictionary[i]=[]
            for i in range(windows_size):
                if(area[i]<self.min_size):
                    continue
                if(width[i]*1.0/height[i]>self.max_ratio or height[i]*1.0/width[i]>self.max_ratio):
                    continue
                label=self.spectral.labels_[i]
                if len(index_dictionary[label])>=self.top_k_in_cluster:
                    continue
                index_dictionary[label].append(i)
            index_list=[]
            for key in index_dictionary:
                index_list.extend(index_dictionary[key])
            
            endtimeInBoxes=time.time()
            logging.info("Cluster get top {} spend {:.3f}".format(self.top_k_in_cluster,endtimeInBoxes-starttimeInBoxes))
            boxes=pd.DataFrame({0:ymins[index_list],1:xmins[index_list],2:ymaxs[index_list],3:xmaxs[index_list].tolist()})
            return boxes
        def detect_image(self,imagefilename):
            starttime = time.time() 
            boxes = self.bing_search.getBoxesOfOneImage(imagefilename,130)
            bing_windows=self.cluster_boxes(boxes)
            #bing_windows=pd.DataFrame({0:ymins,1:xmins,2:ymaxs,3:xmaxs})
            logging.info("Processed bing get {} windows in {:.3f} s.".format(bing_windows.shape[0],time.time() - starttime))
            detections = self.net.detect_windows([(imagefilename,bing_windows.values)])
            #detections = self.net.detect_selective_search([imagefilename])
            df = pd.DataFrame(detections)
            df[COORD_COLS] = pd.DataFrame(
                    data=np.vstack(df['window']),columns=COORD_COLS)
            del(df['window'])
            del(df['filename'])
            predictions_df = pd.DataFrame(np.vstack(df.prediction.values))
            del(df['prediction'])
            midtime = time.time()
            max_val_each=predictions_df.max(1)
            max_ind_each=predictions_df.idxmax(1)
            max_each=pd.concat([max_val_each,max_ind_each],axis=1)
            #max_each=max_each.rename(columns={0:'value',1:'category_id'})
            temp=max_each[max_each[0]>-1.0]
            if(temp.shape[0] == 0):
                max_each=max_each.sort([0],ascending=False).head(1)
            else:
                max_each=temp
            max_each=max_each.join(df,how='inner')
            max_each=max_each.sort([0],ascending=False)
            print max_each
            dets_all=np.vstack(max_each.values)
            dets=self.nms_detections(dets_all,0.1,0.8)
            max_all=max_each.rename(columns={0:'value',1:'category_id',2:'ymin',3:'xmin',4:'ymax',5:'xmax'})
            max_each=pd.DataFrame(dets)
            max_each=max_each.rename(columns={0:'value',1:'category_id',2:'ymin',3:'xmin',4:'ymax',5:'xmax'})
            print max_each
            img=cv2.imread(imagefilename)
            image_size=img.shape[:-1]
            #font=cv2.FONT_ITALIC
            result=[]
            result_all=[]
            index_box=0
            for index , row in max_each.iterrows():
                index_box=index_box+1
                label=self.labels.loc[int(row['category_id']),'name']
                #(xmin,ymin,xmax,ymax,label)=(int(row['xmin']),int(row['ymin']),int(row['xmax']),int(row['ymax']),self.labels.loc[int(row['category_id']),'name'])
                #cv2.rectangle(img,(xmin,ymin),(xmax,ymax),RECTANGLE_COLOR,3)
                result.append((label,row['value'],index_box,row['ymin']/image_size[0],(row['ymax']-row['ymin'])/image_size[0],row['xmin']/image_size[1],(row['xmax']-row['xmin'])/image_size[1]))
                #cv2.putText(img,label,(xmin+10,ymin+10),font,0.5,TEXT_COLOR,1)
            #newimagelist=imagefilename.rsplit('.',1)
            #newimagefilename=newimagelist[0]+'Result.'+newimagelist[1]
            #cv2.imwrite(newimagefilename,img)
            for index , row in max_all.iterrows():
                index_box=index_box+1
                label=self.labels.loc[int(row['category_id']),'name']
                result_all.append((label,row['value'],index_box,row['ymin']/image_size[0],(row['ymax']-row['ymin'])/image_size[0],row['xmin']/image_size[1],(row['xmax']-row['xmin'])/image_size[1]))
            endtime=time.time()
            #print endtime - midtime
            logging.info("Processed {} windows in {:.3f} s.".format(len(detections),endtime-starttime))
            return (True,result,result_all,'%.3f' % (endtime - starttime)) 
            
def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)
    parser.add_option(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=False)

    opts, args = parser.parse_args()
    ImagenetClassifier.default_args.update({'gpu_mode': opts.gpu})
    ImagenetDetection.default_args.update({'gpu_mode':opts.gpu})
    # Initialize classifier
    app.clf = ImagenetClassifier(**ImagenetClassifier.default_args)
    # Initialize detection
    app.det = ImagenetDetection(**ImagenetDetection.default_args)
    if opts.debug:
	app.run(debug=True, host='10.214.34.104', port=opts.port)
    else:
        start_tornado(app, opts.port)

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app)
