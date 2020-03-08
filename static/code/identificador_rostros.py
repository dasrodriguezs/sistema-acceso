from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
from static.code import facenet
from static.code import detect_face
from static.code.constantes import db_config
import os
import pickle
import datetime
import mysql.connector
from static.code.logger_handler import get_logger

logger = get_logger(__name__)


# noinspection SqlNoDataSourceInspection,SqlResolve
class MySQLConector:

    def __init__(self):
        self.mydb = mysql.connector.connect(**db_config)
        self.cursor = self.mydb.cursor()

    def insertar_evento(self, id, tipo, evento, dispositivo, direccion):
        time.strftime('%Y-%m-%d %H:%M:%S')
        query = 'INSERT INTO sistema_acceso.eventos (user_id, tipo_evento, evento, dispositivo, hora, direccion_img) ' \
                'VALUES (%s, %s, %s, %s, %s, %s)'
        logger.debug(query, (id, tipo, evento, dispositivo, datetime.datetime.utcnow(), direccion))
        self.cursor.execute(query, (id, tipo, evento, dispositivo, datetime.datetime.utcnow(), direccion))
        self.mydb.commit()

    def obtener_datos(self, tarjeta_id):
        query = "SELECT * FROM sistema_acceso.usuarios WHERE (id_tarjeta) = '" + tarjeta_id + "'"
        logger.debug(query)
        self.cursor.execute(query)
        return self.cursor.fetchall()[0]

    def cerrar_conexion(self):
        self.cursor.close()
        self.mydb.close()


class Identificador:

    def __init__(self, ubicacion):
        self.img_dir = ubicacion
        model_dir = os.path.join(os.path.dirname(__file__), 'model', '20180402-114759.pb')
        npy = os.path.join(os.path.dirname(__file__), 'npy')
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with self.sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, npy)
                logger.debug('Loading feature extraction model')
                facenet.load_model(model_dir)
                self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

                classifier_filename = os.path.join(os.path.dirname(__file__), 'class', 'classifier_lfw-5-new-1.pkl')
                classifier_filename_exp = os.path.expanduser(classifier_filename)
                with open(classifier_filename_exp, 'rb') as infile:
                    (self.model, self.class_tarjetas) = pickle.load(infile)

    def identify(self, id_tarjeta, tipo, dispositivo):
        mensaje = dict()
        mensaje['list'] = []
        conector = MySQLConector()
        with tf.Graph().as_default():
            with self.sess.as_default():
                minsize = 20  # minimum size of face
                threshold = [0.6, 0.7, 0.7]  # three steps's threshold
                factor = 0.709  # scale factor
                frame_interval = 3
                # batch_size = 1000
                image_size = 182
                input_image_size = 160
                embedding_size = self.embeddings.get_shape()[1]

                lista_tarjetas = [tarjeta for tarjeta in self.class_tarjetas if
                                  not tarjeta.lower().startswith('bounding')]
                lista_tarjetas.sort()
                c = 0
                logger.debug('Start Recognition!')
                frame = cv2.imread(self.img_dir, 0)
                timeF = frame_interval

                if c % timeF == 0:
                    if frame.ndim == 2:
                        frame = facenet.to_rgb(frame)
                    frame = frame[:, :, 0:3]
                    bounding_boxes, _ = detect_face.detect_face(frame, minsize, self.pnet, self.rnet, self.onet,
                                                                threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]
                    logger.debug('Face Detected: %d' % nrof_faces)

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
                                logger.debug('Face is too close')
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

                            predictions = self.model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[
                                np.arange(len(best_class_indices)), best_class_indices]
                            logger.debug('Result Indices: ' + str(best_class_indices[0]))
                            result_names = lista_tarjetas[best_class_indices[0]]
                            logger.debug('Result id: ' + result_names)
                            logger.debug('Probability Result: ' + str(best_class_probabilities[0]))
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20
                            cv2.putText(frame, 'Id: ' + result_names, (text_x, text_y + 30),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (0, 0, 255), thickness=1, lineType=2)
                            cv2.putText(frame, 'Proba: ' + str(best_class_probabilities[0]), (text_x, text_y),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (0, 0, 255), thickness=1, lineType=1)
                            for index in np.argpartition(predictions[0], -4)[-4:]:
                                print('Result id: ' + lista_tarjetas[index] + '\t- Probability Result: ' + str(
                                    predictions[0][index]))
                            if float(best_class_probabilities[0]) > float(0.01):

                                datos = conector.obtener_datos(id_tarjeta)
                                if (datos is not None) and (datos[-1] is not 0) and (result_names == id_tarjeta):
                                    mensaje['list'].append({
                                        'name': datos[1],
                                        'proba': best_class_probabilities[0],
                                        'id_tarjeta': id_tarjeta,
                                        'autorizado': True
                                    })
                                    conector.insertar_evento(datos[0], tipo, 'Exitoso', dispositivo, self.img_dir)
                                elif datos is not None and datos[-1] is 0 and result_names is id_tarjeta:
                                    mensaje['list'].append({
                                        'name': datos[1],
                                        'proba': best_class_probabilities[0],
                                        'id_tarjeta': id_tarjeta,
                                        'autorizado': False
                                    })
                                    conector.insertar_evento(datos[0], tipo, 'Fallido - No Activo', dispositivo,
                                                             self.img_dir)
                            else:
                                datos = conector.obtener_datos(id_tarjeta)
                                conector.insertar_evento(datos[0], tipo, 'Fallido - No Reconocido', dispositivo,
                                                         self.img_dir)
                else:
                    logger.debug(' Unable to align')
            # os.remove(self.img_path)
            new_img = self.img_dir.split('.')
            cv2.imwrite(new_img[0] + 'bb.' + new_img[1], frame)
            logger.debug('Mensaje: ' + str(mensaje))
            conector.cerrar_conexion()
            return mensaje
