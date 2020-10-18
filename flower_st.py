import itertools

import numpy as np
from scipy.spatial.distance import cdist
from PIL import Image
import streamlit as st
import consts

from utils import FlowerArc, load_prec_embs, SaliencyDetection, Classifier


def main(top_k):

    flower_arc = FlowerArc()
    saliency = SaliencyDetection()
    classifier = Classifier()
    st.title("Flower retrieval")
    train_img_fps, train_embs, train_labels = load_prec_embs()
    uploaded_file = st.file_uploader("Choose an image...")

    if uploaded_file is not None:
        st.image(
            uploaded_file,
            caption='Uploaded Image.',
            use_column_width=True
        )
        image = Image.open(uploaded_file)
        img_arr = np.array(image)

        flower_classifier = classifier.classification_predict(img_arr, consts.CLASSIFICATION_THRESHHOLD)

        if (flower_classifier == 0): 
            st.subheader('Flower Detected')

            #saliency
            map_img = saliency.saliency_predict(img_arr)
            bounding_box_img = saliency.bounding_box(img_arr, map_img)
            st.image(bounding_box_img, use_column_width=True, caption='Flower Detection')
            # query emb
            test_emb = flower_arc.predict(img_arr)

            dists = cdist(test_emb, train_embs, metric='euclidean')[0]
            min_dist_indexes = dists.argsort()[:top_k]
            label_indexes = [train_labels[index] + 1 for index in min_dist_indexes]
            img_fps = [train_img_fps[index] for index in min_dist_indexes]

            indices_on_page, images_on_page = \
                map(list, zip(*itertools.islice(zip(label_indexes, img_fps), 0, top_k)))  # noqa
            st.image(images_on_page, width=200, caption=indices_on_page)
        else: 
            st.subheader('Your image doesn\'t contain flowers or flowers are not clear enough to be detected, please try another image')

if __name__ == '__main__':
    main(top_k=12)
