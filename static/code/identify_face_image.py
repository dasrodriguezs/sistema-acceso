from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
from static.code import facenet
from static.code import detect_face
import os
import pickle
import datetime


class identify_face:

    def __init__(self, path):
        self.img_path = path
        modeldir = os.path.join(os.path.dirname(__file__), 'model', '20170511-185253.pb')
        npy = os.path.join(os.path.dirname(__file__), 'npy')
        train_img = os.path.join(os.path.dirname(__file__), '..', 'train_img')
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with self.sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, npy)
                HumanNames = ['daniel.beltran', 'daniel.rodriguez', 'fabian.chacon', 'laura.beltran', 'luis.valderrama',
                              'miguel.ballen']
                # HumanNames = os.listdir(train_img)
                HumanNames.sort()
                print(str(datetime.datetime.utcnow()) + ' Loading feature extraction model')
                facenet.load_model(modeldir)
                self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    @property
    def identify(self):
        mensaje = dict()
        mensaje['list'] = []
        classifier_filename = os.path.join(os.path.dirname(__file__), 'class', 'classifier.pkl')
        train_img = os.path.join(os.path.dirname(__file__), '..', 'train_img')
        with tf.Graph().as_default():
            with self.sess.as_default():
                minsize = 20  # minimum size of face
                threshold = [0.6, 0.7, 0.7]  # three steps's threshold
                factor = 0.709  # scale factor
                margin = 44
                frame_interval = 3
                batch_size = 1000
                image_size = 182
                input_image_size = 160
                HumanNames = ['daniel.beltran', 'daniel.rodriguez', 'fabian.chacon', 'laura.beltran', 'luis.valderrama',
                              'miguel.ballen']
                # HumanNames = os.listdir(train_img)
                HumanNames.sort()
                embedding_size = self.embeddings.get_shape()[1]

                classifier_filename_exp = os.path.expanduser(classifier_filename)
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)

                HumanNames = [name for name in class_names if not name.startswith('Bounding')]
                HumanNames.sort()
                c = 0
                print(str(datetime.datetime.utcnow()) + ' Start Recognition!')
                frame = cv2.imread(self.img_path, 0)
                timeF = frame_interval

                if c % timeF == 0:
                    if frame.ndim == 2:
                        frame = facenet.to_rgb(frame)
                    frame = frame[:, :, 0:3]
                    bounding_boxes, _ = detect_face.detect_face(frame, minsize, self.pnet, self.rnet, self.onet,
                                                                threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]
                    print(str(datetime.datetime.utcnow()) + ' Face Detected: %d' % nrof_faces)

                    if nrof_faces > 0:
                        det = bounding_boxes[:, 0:4]
                        img_size = np.asarray(frame.shape)[0:2]

                        cropped = []
                        scaled = []
                        scaled_reshape = []
                        bb = np.zeros((nrof_faces, 4), dtype=np.int32)
                        for i in range(nrof_faces):
                            emb_array = np.zeros((1, embedding_size))

                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]

                            # inner exception
                            if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                                print(str(datetime.datetime.utcnow()) + ' Face is too close')
                                continue

                            cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                            cropped[i] = facenet.flip(cropped[i], False)
                            scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                            scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size),
                                                   interpolation=cv2.INTER_CUBIC)
                            scaled[i] = facenet.prewhiten(scaled[i])
                            scaled_reshape.append(scaled[i].reshape(-1, input_image_size, input_image_size, 3))
                            feed_dict = {self.images_placeholder: scaled_reshape[i],
                                         self.phase_train_placeholder: False}
                            emb_array[0, :] = self.sess.run(self.embeddings, feed_dict=feed_dict)

                            predictions = model.predict_proba(emb_array)
                            print(str(datetime.datetime.utcnow()) + ' Predictions: ' + str(predictions))
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[
                                np.arange(len(best_class_indices)), best_class_indices]
                            if float(best_class_probabilities[0]) > float(0.5):

                                print(str(datetime.datetime.utcnow()) + ' Result Indices: ', best_class_indices[0])
                                for H_i in HumanNames:
                                    # print(H_i)
                                    if HumanNames[best_class_indices[0]] == H_i:
                                        result_names = HumanNames[best_class_indices[0]]
                                        mensaje['list'].append({
                                            'name': result_names,
                                            'proba': best_class_probabilities[0]
                                        })
                    else:
                        print(str(datetime.datetime.utcnow()) + ' Unable to align')
                # os.remove(self.img_path)
                # cv2.imwrite(filePath, frame)
                print(str(datetime.datetime.utcnow()) + ' ' + str(mensaje))
                return mensaje
