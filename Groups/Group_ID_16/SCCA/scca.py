import keras
from keras              import backend as K
from keras.models       import Sequential, Model
from keras.layers       import TimeDistributed, Dense, Lambda, Reshape
from keras.layers       import dot
from keras.callbacks    import ModelCheckpoint, EarlyStopping
from keras.constraints  import UnitNorm, Constraint
from keras.regularizers import l1

import numpy as np

import scipy.stats
import matplotlib.pyplot as plt

import os
import shutil

def cca_loss(y_true,y_pred):
    return -1*y_pred


class UnitNormWithNonneg(Constraint):
    def __init__(self, nonneg=False, axis=0):
        self.nonneg = nonneg
        self.axis = axis

    def __call__(self, p):
        p = p / (K.epsilon() + K.sqrt(K.sum(K.square(p), axis=self.axis, keepdims=True)))
        if self.nonneg:
            p *= K.cast(p >= 0., K.floatx())

        return p

    def get_config(self):
        return {'name': self.__class__.__name__,
                'nonneg': self.nonneg,
                'axis': self.axis}

def deflate_inputs(current_inputs, original_inputs, u, v):
    Xp      = current_inputs[0][0,:,:]
    Yp      = current_inputs[1][0,:,:]
    X_orig  = original_inputs[0]
    Y_orig  = original_inputs[1]

    qx = np.dot(Xp.T,np.dot(X_orig,u))
    qx = qx / (np.sqrt(np.sum(qx**2)+1e-7))
    Xp = Xp - np.dot(Xp,qx).dot(qx.T)
    X  = Xp.reshape(1,Xp.shape[0],Xp.shape[1])

    qy = np.dot(Yp.T,np.dot(Y_orig,v))
    qy = qy / (np.sqrt(np.sum(qy**2)+1e-7))
    Yp = Yp - np.dot(Yp,qy).dot(qy.T)
    Y  = Yp.reshape(1,Yp.shape[0],Yp.shape[1])

    new_current_inputs = [X,Y]
    return new_current_inputs

def build_scca_model(params):
    dense_vecs  = 1
    
    # X projection model
    modelX = Sequential()
    modelX.add(TimeDistributed(Dense(dense_vecs, use_bias=False,
        kernel_constraint=UnitNormWithNonneg(nonneg=params['nonneg']),
        kernel_regularizer=l1(params['sparsity_X'])),
        input_shape=(params['shape_X'][1],params['shape_X'][2])))

    # Y projection model
    modelY = Sequential()
    modelY.add(TimeDistributed(Dense(dense_vecs, use_bias=False,
        kernel_constraint=UnitNormWithNonneg(nonneg=params['nonneg']),
        kernel_regularizer=l1(params['sparsity_Y'])),
        input_shape=(params['shape_Y'][1],params['shape_Y'][2])))

    # merged model
    merged_model = Sequential()
    merged_output = dot([modelX.output, modelY.output], axes = 1)
    final_model = Model([modelX.input, modelY.input], merged_model(merged_output))

    return final_model

def fit(current_inputs, params):

    orig_inputs = [current_inputs[0][0,:,:].copy(), current_inputs[1][0,:,:].copy()] # save for deflate
        
    model_loss  = cca_loss
    outputs     = np.array([[[0.]]])

    u_comp = []
    v_comp = []
    for i in range(params['nvecs']):

        # build model
        model = build_scca_model(params)
        print(model)

        # compile model
        cbacks = []
        if params['save_best'] is True:
            mc = ModelCheckpoint(filepath='tmp_save.h5',monitor='loss')
            cbacks.append(mc)
        model.compile(optimizer=params['algo'], loss=model_loss)

        # fit model
        model.fit(current_inputs, outputs, epochs=params['its'],
                verbose=params['keras_verbose'])
        # load back checkpoint weights
        if params['save_best'] is True:
            try:
                model.load('tmp_save.h5')
            except:
                pass

        # extract weights and add to component list
        _u_comp     = model.layers[2].get_weights()[0]
        _v_comp     = model.layers[3].get_weights()[0]
        u_comp.append(np.squeeze(_u_comp))
        v_comp.append(np.squeeze(_v_comp))

        # deflate inputs
        current_inputs = deflate_inputs(current_inputs, orig_inputs, _u_comp, _v_comp)


    # make projections
    u_comp  = np.array(u_comp)
    v_comp  = np.array(v_comp)

    return (u_comp, v_comp)

def transform(orig_inputs, u_comp, v_comp):
    x_proj  = np.dot(orig_inputs[0], u_comp.T)
    y_proj  = np.dot(orig_inputs[1], v_comp.T)
    
    return (x_proj, y_proj)

def scca(inmats, nvecs, sparsity=(1e-5,1e-5), its=500,
                algo='nadam', verbose=0, nonneg=False,
                save_best=True):
    # Arguments
    # inmats : A tuple or list (dataset1, dataset2) of shape (N, features)
    # nvecs : The number of dimensions to project onto
    # sparsity: A tuple or list of L1 penalty for dataset1 and dataset2 respectively.
    # its : Number of epochs to train on
    # algo : a string or Keras.optimizer object(which gradient descent algorithm to use)
    # verbose : -1(print nothing), 0(print correlations after each run), 1(print full keras training status)
    # nonneg : boolean (whether the weights of componenets can be non-negetive)
    # Returns
    # (u_comp, v_comp) : tuple of 2D arrays of shape (nvecs, features)
    # (x_proj, y_proj) : tuple of 2D arrays of shape (N, features)

    X = inmats[0]
    Y = inmats[1]
    

    # data must be in three dimensions
    if X.ndim == 2:
        X = X.reshape(1,X.shape[0], X.shape[1])
    if Y.ndim == 2:
        Y = Y.reshape(1, Y.shape[0], Y.shape[1])

    current_inputs  = [X,Y]
    orig_inputs = [current_inputs[0][0,:,:].copy(), current_inputs[1][0,:,:].copy()]

    # set params dictionary
    params = {}
    params['shape_X']       = X.shape
    params['shape_Y']       = Y.shape
    params['sparsity_X']    = sparsity[0]
    params['sparsity_Y']    = sparsity[1]
    params['algo']          = algo
    params['its']           = its
    params['keras_verbose'] = max(0,verbose)
    params['nonneg']        = nonneg
    params['nvecs']         = nvecs
    params['save_best']     = save_best

    # FIT MODEL 
    (u_comp, v_comp) = fit(current_inputs=current_inputs, params=params)
    (x_proj, y_proj) = transform(orig_inputs, u_comp, v_comp)

    return (u_comp, v_comp), (x_proj, y_proj)

