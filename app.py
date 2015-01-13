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
REPO_DIRNAME = os.path.abspath(os.path.dirname(__file__) + '/../..')
UPLOAD_FOLDER = '/tmp/caffe_demos_uploads'
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
    result = app.clf.classify_image(image)
    return flask.render_template(
        'classification.html', has_result=True, result=result, imagesrc=imageurl)

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
    result = app.det.detect_image(filename)
    image = exifutil.open_oriented_im(result[4])
    return flask.render_template(
            'detection.html' , has_result=True, result=result ,imagesrc=imageurl)




@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
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
    result = app.det.detect_image(filename)
    #print result
    image = exifutil.open_oriented_im(result[4])
    return flask.render_template(
            'detection.html' , has_result=True,result=result,
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
            '{}/models/bvlc_reference_caffenet/deploy.prototxt'.format(REPO_DIRNAME)),
        'pretrained_model_file': (
            '{}/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'.format(REPO_DIRNAME)),
        'mean_file': (
            '{}/python/caffe/imagenet/ilsvrc_2012_mean.npy'.format(REPO_DIRNAME)),
        'class_labels_file': (
            '{}/data/ilsvrc12/synset_words.txt'.format(REPO_DIRNAME)),
        'bet_file': (
            '{}/data/ilsvrc12/imagenet.bet.pickle'.format(REPO_DIRNAME)),
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
				'{}/models/bvlc_reference_rcnn_ilsvrc13/deploy.prototxt'.format(REPO_DIRNAME)),
			'pretrained_model_file':(
				'{}/models/bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffemodel'.format(REPO_DIRNAME)),
			'mean_file': (
				'{}/python/caffe/imagenet/ilsvrc_2012_mean.npy'.format(REPO_DIRNAME)),
			'class_labels_file':(
				'{}/data/ilsvrc12/det_synset_words.txt'.format(REPO_DIRNAME)),
	}
	for key , val in default_args.iteritems():
		if not os.path.exists(val):
			raise Exception("File for {} is missing. Should be at: {}".format(key, val))
	default_args['gpu_mode'] = False
        default_args['raw_scale'] = 255
        default_args['image_dim'] = 227
        default_args['channel_swap'] ='2,1,0'
        default_args['context_pad'] = 16

        def __init__(self,model_def_file, pretrained_model_file , mean_file , class_labels_file , gpu_mode , raw_scale ,
                image_dim , channel_swap , context_pad , input_scale = None ):
		logging.info('Loading net and associated files...')
                mean , channel = None , None 
                if mean_file:
                    mean = np.load(mean_file)
                if channel_swap:
                    channel = [ int(s) for s in channel_swap.split(',') ]

                self.net = caffe.Detector(model_def_file,pretrained_model_file,
                        gpu=gpu_mode,mean=mean,input_scale=input_scale,raw_scale=raw_scale,channel_swap=channel,context_pad=context_pad)
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
        def nms_detections(self,dets,overlap=0.1):
            x1 = dets[:,3]
            y1 = dets[:,2]
            x2 = dets[:,5]
            y2 = dets[:,4]
            ind = np.argsort(dets[:,0])
            w = x2 - x1
            h = y2 - y1 
            area = (w*h).astype(float)
            pick=[]
            while len(ind)>0:
                i=ind[-1]
                pick.append(i)
                ind =ind[:-1]
                xx1 = np.maximum(x1[i],x1[ind])
                yy1 = np.maximum(y1[i],y1[ind])
                xx2 = np.minimum(x2[i],x2[ind])
                yy2 = np.minimum(y2[i],y2[ind])
                w = np.maximum(0., xx2 - xx1)
                h = np.maximum(0., yy2 - yy1)
                wh = w*h
                o = wh/(area[i]+area[ind]-wh)
                ind = ind[np.nonzero(o<=overlap)[0]]
            return dets[pick,:]
        def detect_image(self,imagefilename):
            starttime = time.time()
            detections = self.net.detect_selective_search([imagefilename])
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
            max_each=max_each[max_each[0]>0]
            max_each=max_each.join(df,how='inner')
            max_each=max_each.sort([0],ascending=False)
            dets=np.vstack(max_each.values)
            dets=self.nms_detections(dets)
            max_each=pd.DataFrame(dets)
            max_each=max_each.rename(columns={0:'value',1:'category_id',2:'ymin',3:'xmin',4:'ymax',5:'xmax'})
            img=cv2.imread(imagefilename)
            font=cv2.FONT_ITALIC
            result=[]
            for index , row in max_each.iterrows():
                (xmin,ymin,xmax,ymax,label)=(int(row['xmin']),int(row['ymin']),int(row['xmax']),int(row['ymax']),self.labels.loc[int(row['category_id']),'name'])
                cv2.rectangle(img,(xmin,ymin),(xmax,ymax),RECTANGLE_COLOR,3)
                result.append((label,row['value']))
                cv2.putText(img,label,(xmin+10,ymin+10),font,0.5,TEXT_COLOR,1)
            newimagelist=imagefilename.rsplit('.',1)
            newimagefilename=newimagelist[0]+'Result.'+newimagelist[1]
            cv2.imwrite(newimagefilename,img)
            endtime=time.time()
            print endtime - midtime
            logging.info("Processed {} windows in {:.3f} s.".format(len(detections),endtime-starttime))
            return (True,result,result,'%.3f' % (endtime - starttime),newimagefilename) 

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
	app.run(debug=True, host='10.13.94.41', port=opts.port)
    else:
        start_tornado(app, opts.port)

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app)
