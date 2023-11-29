import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
from tensorflow.keras.models import model_from_json

def tao_anh_ngau_nhien(X_test):
    image = np.zeros((10 * 28, 10 * 28), np.uint8)
    data = np.zeros((100, 28, 28, 1), np.uint8)

    for i in range(0, 100):
        n = random.randint(0, 9999)
        sample = X_test[n]
        data[i] = X_test[n]
        x = i // 10
        y = i % 10
        image[x * 28:(x + 1) * 28, y * 28:(y + 1) * 28] = sample[:, :, 0]
    return image, data

def main():
    st.title("Nhận dạng chữ số từ tập dữ liệu MNIST")

    if 'is_load' not in st.session_state:
        # Load model
        model_architecture = 'digit_config.json'
        model_weights = 'digit_weight.h5'
        model = model_from_json(open(model_architecture).read())
        model.load_weights(model_weights)

        OPTIMIZER = tf.keras.optimizers.Adam()
        model.compile(
            loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"]
        )
        st.session_state.model = model

        # Load data
        (_, _), (X_test, y_test) = keras.datasets.mnist.load_data()
        X_test = X_test.reshape((10000, 28, 28, 1))
        X_test = X_test / 255.0
        X_test = X_test.astype('float32')
        st.session_state.X_test = X_test

        st.session_state.is_load = True
        st.write('Lần đầu load model và data')
    else:
        st.write('Đã load model và data rồi')

    if st.button('Tạo ảnh và Nhận dạng'):
        image, data = tao_anh_ngau_nhien(st.session_state.X_test)
        ket_qua = st.session_state.model.predict(data)
        dem = 0
        s = ''
        for x in ket_qua:
            s = s + '%d ' % (np.argmax(x))
            dem = dem + 1
            if (dem % 10 == 0) and (dem < 100):
                s = s + '\n'
        st.image(image)
        st.text(s)

if __name__ == '__main__':
    main()
