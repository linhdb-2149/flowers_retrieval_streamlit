import os
import pickle
import copy
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import grpc
from tensorflow_serving.apis import (
    prediction_service_pb2_grpc,
    predict_pb2
)

from consts import (
    TRAIN_FD,
    TRAIN_PKL_FP,
    TRAIN_LABEL_FP,
    CLASSIFIER_MODEL
)
@st.cache
def load_prec_embs():
    with open(TRAIN_PKL_FP, "rb") as f:
        train_embs = pickle.load(f)

    with open(TRAIN_LABEL_FP, "rb") as f:
        train_labels = pickle.load(f)

    train_img_fps = wfile(TRAIN_FD)
    assert len(train_img_fps) == train_embs.shape[0]

    return train_img_fps, train_embs, train_labels


def wfile(root):
    img_fps = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            img_fps.append(os.path.join(path, name))

    return sorted(img_fps)


class FlowerArc:

    def __init__(self,
                 host="192.168.19.37",
                 port=8700,
                 model_name="flower",
                 model_signature="flower_signature",
                 input_name="input_image",
                 output_name="emb_pred"):

        self.host = host
        self.port = port

        self.channel = grpc.insecure_channel("{}:{}".format(
            self.host, self.port
        ))
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(
            self.channel
        )
        self.input_name = input_name
        self.output_name = output_name

        self.request = predict_pb2.PredictRequest()
        self.request.model_spec.name = model_name
        self.request.model_spec.signature_name = model_signature

    def norm_mean_std(self,
                      img):

        img = img / 255
        img = img.astype('float32')

        mean = np.mean(img, axis=(0, 1, 2))
        std = np.std(img, axis=(0, 1, 2))
        img = (img - mean) / std

        return img

    def test_preprocess(self,
                        img,
                        img_size=(384, 384),
                        expand=True):

        img = cv2.resize(img, img_size)

        # normalize image
        img = self.norm_mean_std(img)

        if expand:
            img = np.expand_dims(img, axis=0)

        return img


    def predict(self, img):

        assert img.ndim == 3

        img = self.test_preprocess(img)

        self.request.inputs[self.input_name].CopyFrom(
            tf.contrib.util.make_tensor_proto(
                img,
                dtype=tf.float32,
                shape=img.shape
            )
        )

        result = self.stub.Predict(self.request, 10.0)

        emb_pred = tf.contrib.util.make_ndarray(
            result.outputs[self.output_name]
        )
        return emb_pred

    

class SaliencyDetection:
    """docstring for SaliencyDetection"""
    def __init__(self,
                 host="192.168.19.37",
                 port=8700,
                 model_name="saliency",
                 model_signature="serving_default",
                 input_image="input_image",
                 pred_mask="pred_mask"):
        self.host = host
        self.port = port

        self.channel = grpc.insecure_channel("{}:{}".format(
            self.host, self.port
        ))
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(
            self.channel
        )
        self.input_image = input_image
        self.pred_mask = pred_mask

        self.request = predict_pb2.PredictRequest()
        self.request.model_spec.name = model_name
        self.request.model_spec.signature_name = model_signature
        
    def saliency_predict(self, img):
        img = cv2.resize(img, (320, 240))
        img = np.expand_dims(img, axis = 0)

        self.request.inputs[self.input_image].CopyFrom(
            tf.contrib.util.make_tensor_proto(
                img,
                dtype=np.float32,
                shape=img.shape
            )
        )

        result = self.stub.Predict(self.request, 10.0)
        pred_mask = tf.contrib.util.make_ndarray(result.outputs[self.pred_mask])[0]
        return pred_mask

    def bounding_box(self, img, map_img_source):

        map_img = copy.deepcopy(map_img_source)
        map_img = map_img.astype(np.float32)
        thres = 0.02
        map_img[map_img >= thres] = 1
        map_img[map_img < thres] = 0

        # crop bbox
        horizontal_indicies = np.where(np.any(map_img, axis=0))[0]
        vertical_indicies = np.where(np.any(map_img, axis=1))[0]

        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        img_arr_2 = copy.deepcopy(img)
        height, width, channels = img.shape
        h_ratio = height / 240
        w_ratio = width / 320
        cv2.rectangle(img_arr_2, (int(x1 * w_ratio), int(y1 * h_ratio)), (int(x2 * w_ratio), int(y2 * h_ratio)), (255,225,0), 4)
        return img_arr_2


class Classifier(object):
    """docstring for ClassName"""
    def __init__(self,
                 host="192.168.19.37",
                 port=8700,
                 model_name="classifier",
                 model_signature="classifier",
                 input_image="input_image",
                 y_pred="y_pred"):
        self.host = host
        self.port = port

        self.channel = grpc.insecure_channel("{}:{}".format(
            self.host, self.port
        ))
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(
            self.channel
        )
        self.input_image = input_image
        self.y_pred = y_pred

        self.request = predict_pb2.PredictRequest()
        self.request.model_spec.name = model_name
        self.request.model_spec.signature_name = model_signature

    def norm_mean_std(self,
                      img):

        img = img / 255
        img = img.astype('float32')

        mean = np.mean(img, axis=(0, 1, 2))
        std = np.std(img, axis=(0, 1, 2))
        img = (img - mean) / std

        return img

    def test_preprocess(self,
                        img,
                        img_size=(384, 384),
                        expand=True):

        img = cv2.resize(img, img_size)

        # normalize image
        img = self.norm_mean_std(img)

        if expand:
            img = np.expand_dims(img, axis=0)

        return img

    def classification_predict(self, img, threshold):
        img = self.test_preprocess(img, img_size=(224, 224),expand=True)
        self.request.inputs[self.input_image].CopyFrom(
            tf.contrib.util.make_tensor_proto(
                img,
                dtype=np.float32,
                shape=img.shape
            )
        )

        result = self.stub.Predict(self.request, 10.0)
        y_pred = tf.contrib.util.make_ndarray(result.outputs[self.y_pred])[0]
        
        if y_pred[0] < threshold: result = 0
        else: result = 1
        return result
