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
import random
from sklearn import cluster
import itertools
from bing import Bing
from bing import Boxes
from enum import Enum

PROJECT_DIRNAME = os.path.abspath(os.path.dirname(__file__))
IMAGE_PREFIX = "/static/temp/"
UPLOAD_FOLDER = PROJECT_DIRNAME + IMAGE_PREFIX
RANDOM_CLASSIFICATION = PROJECT_DIRNAME + "/static/img/classification"
RANDOM_DETECTION = PROJECT_DIRNAME + "/static/img/detection"

ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])
COORD_COLS = ['ymin', 'xmin', 'ymax', 'xmax']
RECTANGLE_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 0, 0)
SQUARE_WAY_ENUM = Enum('upper', 'center', 'under', 'random', 'scale')

# Obtain the flask app object
app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False)


@app.route('/detection')
def detection():
    return flask.render_template('detection.html', has_result=False)


@app.route('/configuration')
def configuration():
    origin_parameter = app.det.get_configuration_parameter()
    display_parameter = app.display.get_display_parameter()
    return flask.render_template('configuration.html',
                                 origin_parameter=origin_parameter,
                                 display_parameter=display_parameter,
                                 SQUARE_WAY_ENUM=SQUARE_WAY_ENUM)


@app.route('/configure_parameter', methods=['POST'])
def configure_parameter():
    origin_parameter = app.det.get_configuration_parameter()
    origin_display_parameter = app.display.get_display_parameter()
    cluster_num_val = int(flask.request.form['cluster_num_val'])
    top_k_in_cluster_val = int(flask.request.form['top_k_in_cluster_val'])
    max_ratio_val = int(flask.request.form['max_ratio_val'])
    min_size_pixel_val = int(flask.request.form['min_size_pixel_val'])
    min_size_percent_val = int(flask.request.form['min_size_percent_val'])
    compression_val = flask.request.form.get('compression')
    if compression_val is None:
        compression_val = False
    else:
        compression_val = True
    max_compression_size_val = int(
        flask.request.form['max_compression_size_val'])
    squareness_val = flask.request.form.get('squareness')
    if squareness_val is None:
        squareness_val = False
    else:
        squareness_val = True
    squareness_way_val = flask.request.form['squareness_way_val']
    display_squareness_val = flask.request.form.get('display_squareness')
    if display_squareness_val is None:
        display_squareness_val = False
    else:
        display_squareness_val = True
    display_width_size_val = int(flask.request.form['display_width_size_val'])
    app.det.set_configuration_parameter(
        cluster_num_val, top_k_in_cluster_val, max_ratio_val,
        min_size_pixel_val, min_size_percent_val, compression_val,
        max_compression_size_val, squareness_val, squareness_way_val)
    app.display.set_display_parameter(display_squareness_val,
                                      display_width_size_val)
    updated_parameter = app.det.get_configuration_parameter()
    updated_display_parameter = app.display.get_display_parameter()
    parameter_len = len(origin_parameter)
    parameter_change = False
    for i in range(parameter_len):
        if origin_parameter[i] != updated_parameter[i]:
            parameter_change = True
            break
    if not parameter_change:
        parameter_len = len(origin_display_parameter)
        for i in range(parameter_len):
            if origin_display_parameter[i] != updated_display_parameter[i]:
                parameter_change = True
                break

    return flask.render_template(
        'configuration_work.html',
        origin_parameter=origin_parameter,
        updated_parameter=updated_parameter,
        origin_display_parameter=origin_display_parameter,
        updated_display_parameter=updated_display_parameter,
        parameter_change=parameter_change)


@app.route('/classification')
def classification():
    return flask.render_template('classification.html', has_result=False)


@app.route('/box_clustering')
def box_clustering():
    return flask.render_template('box_clustering.html', has_result=False)


@app.route('/classify_url', methods=['GET'])
def classify_url():
    imageurl = flask.request.args.get('imageurl', '')
    try:
        string_buffer = StringIO.StringIO(urllib.urlopen(imageurl).read())
        image = caffe.io.load_image(string_buffer)

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
            'classification.html',
            has_result=True,
            result=(False, 'Cannot open image from URL.'))

    logging.info('Image: %s', imageurl)
    filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            imageurl.split('/')[-1]
    filename = os.path.join(UPLOAD_FOLDER, filename_)
    skimage.io.imsave(filename, image)
    logging.info('Saving to %s.', filename)
    result = app.clf.classify_image(image)
    return flask.render_template(
        'classification.html',
        has_result=True,
        result=result,
        imagesrc=embed_image_html(image)  #imagesrc=IMAGE_PREFIX+filename_)
    )


@app.route('/detect_url', methods=['GET'])
def detect_url():
    imageurl = flask.request.args.get('imageurl', '')
    try:
        string_buffer = StringIO.StringIO(urllib.urlopen(imageurl).read())
        image = caffe.io.load_image(string_buffer)
    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
            'detection.html',
            has_result=True,
            result=(False, 'Cannot open image from URL.'))
    logging.info('Image: %s', imageurl)
    image = preprocessing_image(image,
                                (app.det.compress,
                                 app.det.compress_min_height_width,
                                 app.det.need_square, app.det.square_way))
    filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            imageurl.split('/')[-1]
    filename = os.path.join(UPLOAD_FOLDER, filename_)
    skimage.io.imsave(filename, image)
    logging.info('Saving to %s.', filename)
    result = app.det.detect_image(str(filename))
    return flask.render_template(
        'detection.html',
        has_result=True,
        result=result,
        imagesrc=embed_image_html(image)  #imagesrc=IMAGE_PREFIX+filename_)
    )


@app.route('/cluster_url')
def cluster_url():
    imageurl = flask.request.args.get('imageurl', '')
    try:
        string_buffer = StringIO.StringIO(urllib.urlopen(imageurl).read())
        image = caffe.io.load_image(string_buffer)
    except Exception as err:
        logging.info('URL Image open error %s', err)
        return flask.render_template(
            'box_clustering.html',
            has_result=True,
            result=(False, 'Cannot open image from URL.'))
    logging.info('Image: %s', imageurl)
    image = preprocessing_image(image,
                                (app.det.compress,
                                 app.det.compress_min_height_width,
                                 app.det.need_square, app.det.square_way))
    filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            imageurl.split('/')[-1]
    filename = os.path.join(UPLOAD_FOLDER, filename_)
    skimage.io.imsave(filename, image)
    logging.info('Saving to %s', filename)
    result = app.det.cluster_boxes_of_image(str(filename))
    return flask.render_template('box_clustering.html',
                                 has_result=True,
                                 result=result,
                                 imagesrc=embed_image_html(image))


@app.route('/detect_random')
def detect_random():
    index = random.randint(0, len(app.random_detection_list) - 1)
    filename = os.path.join(RANDOM_DETECTION, app.random_detection_list[index])
    (filename, image) = preprocessing_imagefile(
        filename, (app.det.compress, app.det.compress_min_height_width,
                   app.det.need_square, app.det.square_way))
    result = app.det.detect_image(str(filename))
    return flask.render_template('detection.html',
                                 has_result=True,
                                 result=result,
                                 imagesrc=embed_image_html(image))


@app.route('/cluster_random')
def cluster_random():
    index = random.randint(0, len(app.random_detection_list) - 1)
    filename = os.path.join(RANDOM_DETECTION, app.random_detection_list[index])
    (filename, image) = preprocessing_imagefile(
        filename, (app.det.compress, app.det.compress_min_height_width,
                   app.det.need_square, app.det.square_way))
    result = app.det.cluster_boxes_of_image(str(filename))
    return flask.render_template('box_clustering.html',
                                 has_result=True,
                                 result=result,
                                 imagesrc=embed_image_html(image))


@app.route('/classify_random')
def classify_random():
    index = random.randint(0, len(app.random_classification_list) - 1)
    filename = os.path.join(RANDOM_CLASSIFICATION,
                            app.random_classification_list[index])
    image = exifutil.open_oriented_im(filename)
    result = app.clf.classify_image(image)
    return flask.render_template('classification.html',
                                 has_result=True,
                                 result=result,
                                 imagesrc=embed_image_html(image))


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
            'classification.html',
            has_result=True,
            result=(False, 'Cannot open uploaded image.'))

    result = app.clf.classify_image(image)
    return flask.render_template('classification.html',
                                 has_result=True,
                                 result=result,
                                 #imagesrc=IMAGE_PREFIX+filename_
                                 imagesrc=embed_image_html(image))


@app.route('/detect_upload', methods=['POST'])
def detect_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
                werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)
        #image = exifutil.open_oriented_im(filename)
        #logging.info(image)
    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'detection.html',
            has_result=True,
            result=(False, 'Cannot open uploaded image.'))
    (filename, image) = preprocessing_imagefile(
        filename, (app.det.compress, app.det.compress_min_height_width,
                   app.det.need_square, app.det.square_way))
    result = app.det.detect_image(str(filename))
    return flask.render_template('detection.html',
                                 has_result=True,
                                 result=result,
                                 imagesrc=embed_image_html(image))


@app.route('/cluster_upload', methods=['POST'])
def cluster_upload():
    try:
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
                werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)
    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'box_clustering.html',
            has_result=True,
            result=(False, 'Cannot open uploaded image.'))
    (filename, image) = preprocessing_imagefile(
        filename, (app.det.compress, app.det.compress_min_height_width,
                   app.det.need_square, app.det.square_way))
    result = app.det.cluster_boxes_of_image(str(filename))
    return flask.render_template('box_clustering.html',
                                 has_result=True,
                                 result=result,
                                 imagesrc=embed_image_html(image))


@app.route('/detect_local', methods=['GET'])
def detect_local():
    filename_ = flask.request.args.get('imagefilename', '')
    if not allowed_file(filename_):
        logging.info('Detect Local Image Error, not an image file')
        return flask.render_template('detection.html',
                                     has_result=True,
                                     result=(False, 'Detect error image'))
    imagefilename = PROJECT_DIRNAME + "/static/" + filename_
    if not os.path.isfile(imagefilename):
        logging.info('Detect Local Image Error,%s', imagefilename)
        return flask.render_template('detection.html',
                                     has_result=True,
                                     result=(False, 'Detect image path error'))
    (filename, image) = preprocessing_imagefile(
        imagefilename, (app.det.compress, app.det.compress_min_height_width,
                        app.det.need_square, app.det.square_way))
    result = app.det.detect_image(str(filename))
    return flask.render_template('detection.html',
                                 has_result=True,
                                 result=result,  #imagesrc="/static/"+filename_
                                 imagesrc=embed_image_html(image))


@app.route('/cluster_local', methods=['GET'])
def cluster_local():
    filename_ = flask.request.args.get('imagefilename', '')
    if not allowed_file(filename_):
        logging.info('Cluster Boxes of Local Image Error, not an image file')
        return flask.render_template(
            'box_clustering.html',
            has_result=True,
            result=(False, 'Cluster boxes of error image'))
    imagefilename = PROJECT_DIRNAME + "/static/" + filename_
    if not os.path.isfile(imagefilename):
        logging.info('Cluster Boxes of Local Image Error: %s', imagefilename)
        return flask.render_template(
            'box_clustering.html',
            has_result=True,
            result=(False, 'Cluster boxes of error image'))
    (filename, image) = preprocessing_imagefile(
        imagefilename, (app.det.compress, app.det.compress_min_height_width,
                        app.det.need_square, app.det.square_way))
    result = app.det.cluster_boxes_of_image(str(filename))
    return flask.render_template('box_clustering.html',
                                 has_result=True,
                                 result=result,
                                 imagesrc=embed_image_html(image))


@app.route('/classify_local', methods=['GET'])
def classify_local():
    filename_ = flask.request.args.get('imagefilename', '')
    if not allowed_file(filename_):
        logging.info('Classify Local Image Error, not an image file')
        return flask.render_template('classification.html',
                                     has_result=True,
                                     result=(False, 'Classify error image'))
    imagefilename = PROJECT_DIRNAME + "/static/" + filename_
    if not os.path.isfile(imagefilename):
        logging.info('Classify Local Image Error,%s', imagefilename)
        return flask.render_template(
            'classification.html',
            has_result=True,
            result=(False, 'Classify image path error'))
    image = exifutil.open_oriented_im(imagefilename)
    result = app.clf.classify_image(image)
    return flask.render_template('classification.html',
                                 has_result=True,
                                 result=result,  #imagesrc="/static/"+filename_
                                 imagesrc=embed_image_html(image))


def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = PILImage.fromarray((255 * image).astype('uint8'))
    (width,height)=image_pil.size
    (display_squareness,display_width_size) = app.display.get_display_parameter()
    if display_squareness:
        image_pil = image_pil.resize((display_width_size,display_width_size))
    else:
        image_pil = image_pil.resize((display_width_size,int(height*1.0/width*display_width_size)))
    (width,height)=image_pil.size
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return ('data:image/png;base64,' + data , height , width ) 


def allowed_file(filename):
    return ('.' in filename and
            filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS)


def do_compression_image(image_maxtrix, compress_min_height_width):
    (height, width) = image_maxtrix.shape[:-1]
    min_between = min(height, width)
    if min_between < compress_min_height_width:
        logging.info(
            "The image minimum between height and width is less than requirement,Don't need compression")
        return image_maxtrix
    image_maxtrix = caffe.io.resize_image(image_maxtrix, (
        int(height * 1.0 / min_between * compress_min_height_width),
        int(width * 1.0 / min_between * compress_min_height_width)))
    return image_maxtrix


def do_squareness_image(image_maxtrix, square_way):
    (height, width) = image_maxtrix.shape[:-1]
    if (height > width):
        if square_way == SQUARE_WAY_ENUM.upper:
            image_maxtrix = image_maxtrix[0:width,::]
        elif square_way == SQUARE_WAY_ENUM.center:
            image_maxtrix = image_maxtrix[(height - width) / 2:(height + width)
                                          / 2,::]
        elif square_way == SQUARE_WAY_ENUM.under:
            image_maxtrix = image_maxtrix[height - width:height,::]
        elif square_way == SQUARE_WAY_ENUM.random:
            random_height = random.randint(0, height - width)
            image_maxtrix = image_maxtrix[random_height:random_height + width,::]
        else:
            image_maxtrix = caffe.io.resize_image(image_maxtrix,
                                                  (width, width))
    else:
        if square_way == SQUARE_WAY_ENUM.upper:
            image_maxtrix = image_maxtrix[:, 0:height,:]
        elif square_way == SQUARE_WAY_ENUM.center:
            image_maxtrix = image_maxtrix[:, (width - height) / 2:
                                          (width + height) / 2,:]
        elif square_way == SQUARE_WAY_ENUM.under:
            image_maxtrix = image_maxtrix[:, width - height:width,:]
        elif square_way == SQUARE_WAY_ENUM.random:
            random_width = random.randint(0, width - height)
            image_maxtrix = image_maxtrix[:, random_width:random_width +
                                          height,:]
        else:
            image_maxtrix = caffe.io.resize_image(image_maxtrix,
                                                  (height, height))
    return image_maxtrix


def preprocessing_image(image_maxtrix, operation):
    (compress, compress_min_height_width, need_square, square_way) = operation
    if compress:
        image_maxtrix = do_compression_image(image_maxtrix,
                                             compress_min_height_width)
        logging.info("The Compression Operation is Finished")
    else:
        logging.info("The Compression Operation is False,Pass")
    if need_square:
        image_maxtrix = do_squareness_image(image_maxtrix, square_way)
        logging.info("The Squareness Operation is Finished")
    else:
        logging.info("The Squareness Operation is False,Pass")
    return image_maxtrix


def preprocessing_imagefile(imagefilename, operation):
    filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            imagefilename.split('/')[-1]
    filename = os.path.join(UPLOAD_FOLDER, filename_)
    image_maxtrix = caffe.io.load_image(imagefilename)
    image_maxtrix = preprocessing_image(image_maxtrix, operation)
    skimage.io.imsave(filename, image_maxtrix)
    return (filename, image_maxtrix)


class ImagenetClassifier(object):
    default_args = {
        'model_def_file': (
            '{}/models/bvlc_reference_caffenet/deploy.prototxt'.format(
                PROJECT_DIRNAME)
        ),
        'pretrained_model_file':
        ('{}/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'.format(
            PROJECT_DIRNAME)),
        'mean_file': (
            '{}/models/ilsvrc12/ilsvrc_2012_mean.npy'.format(PROJECT_DIRNAME)
        ),
        'class_labels_file': (
            '{}/models/ilsvrc12/synset_words.txt'.format(PROJECT_DIRNAME)
        ),
        'bet_file': (
            '{}/models/ilsvrc12/imagenet.bet.pickle'.format(PROJECT_DIRNAME)
        ),
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
        self.net = caffe.Classifier(model_def_file, pretrained_model_file,
                                    image_dims=(image_dim, image_dim),
                                    raw_scale=raw_scale,
                                    mean=np.load(mean_file),
                                    channel_swap=(2, 1, 0),
                                    gpu=gpu_mode)

        with open(class_labels_file) as f:
            labels_df = pd.DataFrame([{
                'synset_id': l.strip().split(' ')[0],
                'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
            } for l in f.readlines()])
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
            meta = [(p, '%.5f' % scores[i])
                    for i, p in zip(indices, predictions)]
            expected_infogain = np.dot(self.bet['probmat'],
                                       scores[self.bet['idmapping']])
            expected_infogain *= self.bet['infogain']
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
        'model_def_file': (
            '{}/models/bvlc_reference_rcnn_ilsvrc13/deploy.prototxt'.format(
                PROJECT_DIRNAME)
        ),
        'pretrained_model_file':
        ('{}/models/bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffemodel'.format(
            PROJECT_DIRNAME)),
        'mean_file': (
            '{}/models/ilsvrc12/ilsvrc_2012_mean.npy'.format(PROJECT_DIRNAME)
        ),
        'class_labels_file': (
            '{}/models/ilsvrc12/det_synset_words.txt'.format(PROJECT_DIRNAME)
        ),
    }
    for key, val in default_args.iteritems():
        if not os.path.exists(val):
            raise Exception(
                "File for {} is missing. Should be at: {}".format(key, val))
    default_args['bing_model'] = '{}/models/bing/ObjNessB2W8MAXBGR'.format(
        PROJECT_DIRNAME)
    default_args['gpu_mode'] = False
    default_args['raw_scale'] = 255
    default_args['image_dim'] = 227
    default_args['channel_swap'] = '2,1,0'
    default_args['context_pad'] = 16
    default_args['cluster_num'] = 10
    default_args['top_k_in_cluster'] = 10
    default_args['max_ratio'] = 4
    default_args['min_size'] = 10
    default_args['min_size_percent'] = 100
    default_args['compress'] = True
    default_args['compress_min_height_width'] = 512
    default_args['need_square'] = False
    default_args['square_way'] = SQUARE_WAY_ENUM[3]

    def get_configuration_parameter(self):
        return (self.cluster_num, self.top_k_in_cluster, self.max_ratio,
                self.min_size, self.min_size_percent, self.compress,
                self.compress_min_height_width, self.need_square,
                self.square_way)

    def set_configuration_parameter(self, cluster_num, top_k_in_cluster,
                                    max_ratio, min_size, min_size_percent,
                                    compress, compress_min_height_width,
                                    need_square, square_way):
        logging.info("Cluster Number change from {:} to {:}".format(
            self.cluster_num, cluster_num))
        self.cluster_num = cluster_num
        logging.info("Top K in cluster change from {:} to {:}".format(
            self.top_k_in_cluster, top_k_in_cluster))
        self.top_k_in_cluster = top_k_in_cluster
        logging.info(
            "Max Ratio between width and height change from {:} to {:}".format(
                self.max_ratio, max_ratio))
        self.max_ratio = max_ratio
        logging.info(
            "Min Size of the square of pixel change from {:} to {:}".format(
                self.min_size, min_size))
        self.min_size = min_size
        logging.info("Min Size Percent change from {:} to {:}".format(
            self.min_size_percent, min_size_percent))
        self.min_size_percent = min_size_percent
        self.spectral = cluster.SpectralClustering(n_clusters=cluster_num,
                                                   affinity='precomputed')
        logging.info("Image Compression change from {:} to {:}".format(
            self.compress, compress))
        self.compress = compress
        if self.compress == True:
            logging.info(
                "The max size of the minimum between height and width change from {:} to {:}".format(
                    self.compress_min_height_width, compress_min_height_width))
            self.compress_min_height_width = compress_min_height_width
        logging.info(
            "Wheather image need square operation change from {:} to {:}".format(
                self.need_square, need_square))
        self.need_square = need_square
        if self.need_square == True:
            logging.info(
                "The way of square operation change from {:} to {:}".format(
                    self.square_way, square_way))
            for one_way in SQUARE_WAY_ENUM:
                if str(one_way) == square_way:
                    self.square_way = one_way

    def __init__(self, model_def_file, pretrained_model_file, mean_file,
                 class_labels_file, bing_model, gpu_mode, raw_scale, image_dim,
                 channel_swap, context_pad, cluster_num, top_k_in_cluster,
                 max_ratio, min_size, min_size_percent, compress,
                 compress_min_height_width, need_square, square_way,
                 input_scale=None):
        logging.info('Loading net and associated files...')
        mean, channel = None, None
        if mean_file:
            mean = np.load(mean_file)
        if channel_swap:
            channel = [int(s) for s in channel_swap.split(',')]

        self.net = caffe.Detector(model_def_file, pretrained_model_file,
                                  gpu=gpu_mode,
                                  mean=mean,
                                  input_scale=input_scale,
                                  raw_scale=raw_scale,
                                  channel_swap=channel,
                                  context_pad=context_pad)
        self.bing_search = Bing(2, 8, 2)
        self.bing_search.loadTrainModel(bing_model)
        if gpu_mode:
            print 'GPU mode'
        else:
            print 'CPU mode'
        with open(class_labels_file) as f:
            labels_df = pd.DataFrame([{
                'synset_id': l.strip().split(' ')[0],
                'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
            } for l in f.readlines()])
        self.labels = labels_df.sort('synset_id')
        self.cluster_num = cluster_num
        self.top_k_in_cluster = top_k_in_cluster
        self.max_ratio = max_ratio
        self.min_size = min_size
        self.min_size_percent = min_size_percent
        self.min_pixel_by_min = self.min_size
        self.compress = compress
        self.compress_min_height_width = compress_min_height_width
        self.need_square = need_square
        self.square_way = square_way
        self.spectral = cluster.SpectralClustering(n_clusters=cluster_num,
                                                   affinity='precomputed')

    def removeIOUandOverlap(self, i, index, x1, y1, x2, y2, area, iou,
                            overlap,label):
        xx1 = np.maximum(x1[i], x1[index])
        yy1 = np.maximum(y1[i], y1[index])
        xx2 = np.minimum(x2[i], x2[index])
        yy2 = np.minimum(y2[i], y2[index])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / (area[i] + area[index] - wh)
        oo = wh / np.minimum(area[i], area[index])
        index_new = []
        for one in range(len(index)):
            if(o[one]<=iou)and(oo[one]<=overlap):
                index_new.append(index[one])
            elif (label!=None)and(label[index[one]]!=label[i]):
                index_new.append(index[one])
        return index_new
        '''
        first_match = np.nonzero(o <= iou)[0]
        second_match = np.nonzero(oo <= overlap)[0]
        index_new = index[np.intersect1d(second_match, first_match)]
        if label != None:
            index_diff = np.setdiff1d ( index , index_new ) 
            print index_diff [ np.nonzero ( (label[index_diff] != label [i]) )[0] ]
        
        return index_new
        '''

    def nms_detections(self, dets, iou=0.1, overlap=0.8):
        x1 = dets[:, 3]
        y1 = dets[:, 2]
        x2 = dets[:, 5]
        y2 = dets[:, 4]
        label = dets [:,1]
        ind = np.argsort(dets[:, 0])
        dets_len = len(ind)
        threshold = -0.2
        count = 0
        ok = False
        for i in ind:
            if dets[i, 0] >= threshold:
                ind_one = ind[:count]
                ind_two = ind[count:]
                ok = True
                break
            count += 1
        if ok == False:
            ind_one = ind[:]
            ind_two = []

            #pick=ind[:].tolist()[::-1]
            #return dets[pick,:]

            #if(dets_len <=1):
            #    pick=ind[:].tolist()[::-1]
            #    return dets[pick,:]
            #else:
            #    pick=ind[-1:].tolist()[::-1]
        w = x2 - x1
        h = y2 - y1
        area = (w * h).astype(float)
        #ind=ind[:-1]
        pick = []
        #pick=ind[:].tolist()[::-1]
        while len(ind_two) > 0:
            i = ind_two[-1]
            pick.append(i)
            ind_two = ind_two[:-1]
            ind_two = self.removeIOUandOverlap(i, ind_two, x1, y1, x2, y2,
                                               area, iou * 3, overlap, label)
            ind_one = self.removeIOUandOverlap(i, ind_one, x1, y1, x2, y2,
                                               area, iou, overlap , None )
        while len(ind_one) > 0:
            i = ind_one[-1]
            pick.append(i)
            ind_one = ind_one[:-1]
            ind_one = self.removeIOUandOverlap(i, ind_one, x1, y1, x2, y2,
                                               area, iou, overlap , None )
        return dets[pick,:]

    def get_box_of_each_cluster_boxes(self, boxes):
        ymins = np.array([s for s in boxes.ymins()]).astype(int)
        ymaxs = np.array([s for s in boxes.ymaxs()]).astype(int)
        xmins = np.array([s for s in boxes.xmins()]).astype(int)
        xmaxs = np.array([s for s in boxes.xmaxs()]).astype(int)
        #bing_windows=pd.DataFrame({0:ymins,1:xmins,2:ymaxs,3:xmaxs})
        #return bing_windows
        windows_size = len(xmins)
        width = xmaxs - xmins
        height = ymaxs - ymins
        area = (width * height).astype(float)
        distances = np.zeros((windows_size, windows_size))
        for i in range(windows_size):
            xx1 = np.maximum(xmins[i], xmins)
            yy1 = np.maximum(ymins[i], ymins)
            xx2 = np.minimum(xmaxs[i], xmaxs)
            yy2 = np.minimum(ymaxs[i], ymaxs)
            w = np.maximum(0., xx2 - xx1)
            h = np.maximum(0., yy2 - yy1)
            wh = w * h
            distances[i] = wh / (area[i] + area - wh)
        starttimeInBoxes = time.time()
        self.spectral.fit(distances)
        endtimeInBoxes = time.time()
        logging.info(
            "Cluster speend {:.3f}".format(endtimeInBoxes - starttimeInBoxes))
        index_dictionary = {}
        for i in range(self.cluster_num):
            index_dictionary[i] = []
        for i in range(windows_size):
            if (area[i] < self.min_pixel_by_min):
                continue
            if (width[i] * 1.0 / height[i] > self.max_ratio or
                height[i] * 1.0 / width[i] > self.max_ratio):
                continue
            label = self.spectral.labels_[i]
            if len(index_dictionary[label]) >= self.top_k_in_cluster:
                continue
            index_dictionary[label].append(i)
        return (index_dictionary, ymins, ymaxs, xmins, xmaxs)

    def cluster_boxes(self, boxes):
        index_list = []
        (index_dictionary, ymins, ymaxs, xmins,
         xmaxs) = self.get_box_of_each_cluster_boxes(boxes)
        for key in index_dictionary:
            index_list.extend(index_dictionary[key])
        boxes = pd.DataFrame({
            0: ymins[index_list],
            1: xmins[index_list],
            2: ymaxs[index_list],
            3: xmaxs[index_list].tolist()
        })
        return boxes

    def detect_image(self, imagefilename):
        starttime = time.time()
        img = cv2.imread(imagefilename)
        image_size = img.shape[:-1]
        self.min_pixel_by_min = min(self.min_size * self.min_size,
                                    self.min_size_percent * 0.01 *
                                    image_size[0] * image_size[1])
        boxes = self.bing_search.getBoxesOfOneImage(imagefilename, 130)
        bing_windows = self.cluster_boxes(boxes)
        #bing_windows=pd.DataFrame({0:ymins,1:xmins,2:ymaxs,3:xmaxs})
        logging.info("Processed bing get {} windows in {:.3f} s.".format(
            bing_windows.shape[0], time.time() - starttime))
        detections = self.net.detect_windows([(imagefilename,
                                               bing_windows.values)])
        #detections = self.net.detect_selective_search([imagefilename])
        df = pd.DataFrame(detections)
        df[COORD_COLS] = pd.DataFrame(data=np.vstack(df['window']),
                                      columns=COORD_COLS)
        del (df['window'])
        del (df['filename'])
        predictions_df = pd.DataFrame(np.vstack(df.prediction.values))
        del (df['prediction'])
        midtime = time.time()
        max_val_each = predictions_df.max(1)
        max_ind_each = predictions_df.idxmax(1)
        max_each = pd.concat([max_val_each, max_ind_each], axis=1)
        #max_each=max_each.rename(columns={0:'value',1:'category_id'})
        temp = max_each[max_each[0] > -1.0]
        if (temp.shape[0] == 0):
            max_each = max_each.sort([0], ascending=False).head(1)
        else:
            max_each = temp
        max_each = max_each.join(df, how='inner')
        max_each = max_each.sort([0], ascending=False)
        print max_each
        dets_all = np.vstack(max_each.values)
        dets = self.nms_detections(dets_all, 0.1, 0.8)
        max_all = max_each.rename(columns={
            0: 'value',
            1: 'category_id',
            2: 'ymin',
            3: 'xmin',
            4: 'ymax',
            5: 'xmax'
        })
        max_each = pd.DataFrame(dets)
        max_each = max_each.rename(columns={
            0: 'value',
            1: 'category_id',
            2: 'ymin',
            3: 'xmin',
            4: 'ymax',
            5: 'xmax'
        })
        print max_each
        #font=cv2.FONT_ITALIC
        result = []
        result_all = []
        index_box = 0
        for index, row in max_each.iterrows():
            index_box = index_box + 1
            label = self.labels.loc[int(row['category_id']), 'name']
            #(xmin,ymin,xmax,ymax,label)=(int(row['xmin']),int(row['ymin']),int(row['xmax']),int(row['ymax']),self.labels.loc[int(row['category_id']),'name'])
            #cv2.rectangle(img,(xmin,ymin),(xmax,ymax),RECTANGLE_COLOR,3)
            result.append((label, row['value'], index_box, row['ymin'] /
                           image_size[0], (row['ymax'] - row['ymin']) /
                           image_size[0], row['xmin'] / image_size[1],
                           (row['xmax'] - row['xmin']) / image_size[1]))
            #cv2.putText(img,label,(xmin+10,ymin+10),font,0.5,TEXT_COLOR,1)
            #newimagelist=imagefilename.rsplit('.',1)
            #newimagefilename=newimagelist[0]+'Result.'+newimagelist[1]
            #cv2.imwrite(newimagefilename,img)
        now_index_box = index_box
        for index, row in max_all.iterrows():
            index_box = index_box + 1
            label = self.labels.loc[int(row['category_id']), 'name']
            result_all.append((label, row['value'], index_box, row['ymin'] /
                               image_size[0], (row['ymax'] - row['ymin']) /
                               image_size[0], row['xmin'] / image_size[1],
                               (row['xmax'] - row['xmin']) / image_size[1]))
        endtime = time.time()
        #print endtime - midtime
        logging.info("Processed {} windows in {:.3f} s.".format(
            len(detections), endtime - starttime))
        return (True, result, result_all, '%.3f' %
                    (endtime - starttime))
    def cluster_boxes_of_image(self, imagefilename):
        starttime = time.time()
        boxes = self.bing_search.getBoxesOfOneImage(imagefilename, 130)
        img = cv2.imread(imagefilename)
        image_size = img.shape[:-1]
        self.min_pixel_by_min = min(self.min_size * self.min_size,
                                    self.min_size_percent * 0.01 *
                                    image_size[0] * image_size[1])
        (index_dictionary, ymins, ymaxs, xmins,
         xmaxs) = self.get_box_of_each_cluster_boxes(boxes)
        cluster_list = []
        for index_of_cluster in index_dictionary:
            each_cluster = []
            index_in_cluster = []
            for value in index_dictionary[index_of_cluster]:
                r_color = random.randint(0, 255)
                g_color = random.randint(0, 255)
                b_color = random.randint(0, 255)
                each_cluster.append((value,\
                            ymins[value] * 1.0 / image_size[0], (ymaxs[value] - ymins[value]) * 1.0 / image_size[0],\
                            xmins[value] * 1.0 / image_size[1], (xmaxs[value] - xmins[value]) * 1.0 / image_size[1], \
                            r_color, g_color, b_color))
                index_in_cluster.append(value)
            cluster_list.append(
                (index_of_cluster, each_cluster, index_in_cluster))
        endtime = time.time()
        return (True, cluster_list, '%.3f' % (endtime - starttime))

class Display(object):
    default_args = dict()
    default_args["display_squareness"] = False
    default_args["display_width_size"] = 512

    def __init__(self, display_squareness, display_width_size):
        logging.info("Loading display parameter")
        self.display_squareness = display_squareness
        self.display_width_size = display_width_size

    def get_display_parameter(self):
        return (self.display_squareness, self.display_width_size)

    def set_display_parameter(self, display_squareness, display_width_size):
        logging.info(
            "Squaring image before display change from {:} to {:}".format(
                self.display_squareness, display_squareness))
        self.display_squareness = display_squareness
        logging.info(
            "The width size of image to display change from {:} to {:}".format(
                self.display_width_size, display_width_size))
        self.display_width_size = display_width_size


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
    parser.add_option('-d', '--debug',
                      help="enable debug mode",
                      action="store_true",
                      default=False)
    parser.add_option('-p', '--port',
                      help="which port to serve content on",
                      type='int',
                      default=5000)
    parser.add_option('-g', '--gpu',
                      help="use gpu mode",
                      action='store_true',
                      default=False)

    opts, args = parser.parse_args()
    ImagenetClassifier.default_args.update({'gpu_mode': opts.gpu})
    ImagenetDetection.default_args.update({'gpu_mode': opts.gpu})
    # Initialize classifier
    app.clf = ImagenetClassifier(**ImagenetClassifier.default_args)
    # Initialize detection
    app.det = ImagenetDetection(**ImagenetDetection.default_args)
    app.display = Display(**Display.default_args)
    app.random_detection_list = os.listdir(RANDOM_DETECTION)
    app.random_classification_list = os.listdir(RANDOM_CLASSIFICATION)
    if opts.debug:
        app.run(debug=True, host='10.214.34.104', port=opts.port)
    else:
        start_tornado(app, opts.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app)
