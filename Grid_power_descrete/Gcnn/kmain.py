import unittest
import os
import tempfile
import numpy as np
from keras.models import Sequential
from keras_gcn.backend import keras
from keras_gcn import GraphConv
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, GlobalAveragePooling2D

class TestGraphConv(unittest.TestCase):

    input_data = [
        [
            [0, 1, 2],
            [2, 5, 4],
            [4, 8, 6],
            [7, 10, 8],
        ]
    ]
    input_edge = [
        [
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    ]

    def test_average_step_1(self):
        data_layer = keras.layers.Input(shape=(None, 3), name='Input-Data')
        edge_layer = keras.layers.Input(shape=(None, None), dtype='int32', name='Input-Edge')
        conv_layer = GraphConv(
            units=5,
            step_num=1,
            kernel_initializer='ones',
            bias_initializer='ones',
            name='GraphConv',
        )([data_layer, edge_layer])
        
        model = keras.models.Model(inputs=[data_layer, edge_layer], outputs=conv_layer)
        model.compile(
            optimizer='adam',
            loss='mae',
            metrics=['mae'],
        )
        model.summary()
        print([self.input_data, self.input_edge])
        predicts = model.predict([self.input_data, self.input_edge])[0]
        print("this is prediction",predicts)


    def inter_act(self):
        data_layer = keras.layers.Input(shape=(None, 3), name='Input-Data')
        edge_layer = keras.layers.Input(shape=(None, None), dtype='int32', name='Input-Edge')
        model = Sequential()
        model.add(GraphConv(
            units=5,
            step_num=1,
            kernel_initializer='ones',
            bias_initializer='ones',
            name='GraphConv',
        ))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(60))
        model.add(Activation('relu'))
        model.add(Dense(2))
        predicts = model.predict([self.input_data, self.input_edge])[0]
    def test_average_step_inf(self):
        data_layer = keras.layers.Input(shape=(None, 3), name='Input-Data')
        edge_layer = keras.layers.Input(shape=(None, None), dtype='int32', name='Input-Edge')
        conv_layer = GraphConv(
            units=2,
            step_num=60000000,
            kernel_initializer='ones',
            use_bias=False,
            bias_initializer='ones',
            name='GraphConv',
        )([data_layer, edge_layer])
        model = keras.models.Model(inputs=[data_layer, edge_layer], outputs=conv_layer)
        model.compile(
            optimizer='adam',
            loss='mae',
            metrics=['mae'],
        )
        model_path = os.path.join(tempfile.gettempdir(), 'test_save_load_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={'GraphConv': GraphConv})
        predicts = model.predict([self.input_data, self.input_edge])[0].tolist()
        expects = np.asarray([
            [9., 9.],
            [9., 9.],
            [9., 9.],
            [22., 22.],
        ])
        self.assertTrue(np.allclose(expects, predicts), predicts)

    def test_fit(self):
        data_layer = keras.layers.Input(shape=(None, 3), name='Input-Data')
        edge_layer = keras.layers.Input(shape=(None, None), dtype='int32', name='Input-Edge')
        conv_layer = GraphConv(
            units=2,
            name='GraphConv',
        )([data_layer, edge_layer])
        model = keras.models.Model(inputs=[data_layer, edge_layer], outputs=conv_layer)
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mean_squared_error'],
        )
        expects = np.asarray([[
            [9.5, 0.7],
            [6.5, 0.7],
            [9.5, 0.7],
            [22.8, 1.0],
        ]])
        model.fit(
            x=[self.input_data, self.input_edge],
            y=expects,
            epochs=10000,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='loss', patience=5),
            ],
            verbose=False,
        )
        predicts = model.predict([self.input_data, self.input_edge])
        self.assertTrue(np.allclose(expects, predicts, rtol=0.1, atol=0.1), predicts)

test=TestGraphConv()
#test.test_average_step_1()
test.inter_act()
