import tensorflow_addons as tfa
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import glob
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import Input, Conv1D,Conv2D, Activation, MaxPool2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense,concatenate, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, ZeroPadding2D, LeakyReLU, ReLU, AveragePooling2D,GlobalAveragePooling2D
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import models, optimizers, regularizers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix,multilabel_confusion_matrix
from skimage.io import imshow
from tensorflow.keras import layers
from tensorflow.keras.applications import *
from skimage.io import imread,imshow,imsave
from keras.layers import MaxPool1D,Conv3D, MaxPool3D, Flatten, Dense,MaxPool2D,GlobalAveragePooling1D
from tensorflow import keras
from tensorflow.keras.layers import  Add,Input, Conv1D, BatchNormalization, Activation, Concatenate, AveragePooling1D, Flatten, Dense,GlobalAveragePooling1D,RandomCrop
from tensorflow.keras.models import Model
import cv2
import tensorflow.compat.v1 as tf
import os
import random
import pandas as pd
import os.path
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from sklearn import preprocessing
from scipy.signal import butter, lfilter,find_peaks
from tensorflow.keras.initializers import GlorotNormal
from natsort import natsorted
from keras import backend as K
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
def normalize(input_list):
    x_nor = np.copy(input_list)
    for n,signal in enumerate(input_list):
        x_max = np.max(signal)
        x_min = np.min(signal)
        x_mean = np.mean(signal)
        x_nor[n] = (signal-x_min)/(x_max-x_min)
    return x_nor
def load_npz_data(fpath):
    all_data = np.load(fpath)
    return all_data["arr_0"]
def SE_Block(input_tensor,ratio = 16):
    input_shape = K.int_shape(input_tensor)
    squeeze = tf.keras.layers.GlobalAveragePooling1D()(input_tensor)
    excitation = tf.keras.layers.Dense(units = input_shape[-1]//ratio, kernel_initializer='he_normal',activation='relu')(squeeze)
    excitation = tf.keras.layers.Dense(units = input_shape[-1],activation='sigmoid')(excitation)
    #excitation = tf.reshape(excitation, [-1, 1, input_shape[-1]])
    scale = tf.keras.layers.Multiply()([input_tensor, excitation])
    return scale


s0=sorted(glob.glob("C:/Users/user/Desktop/PPT/ICBHI/Phase1摘要/class/fmri_p1_data/happy/*"))
s1=sorted(glob.glob("C:/Users/user/Desktop/PPT/ICBHI/Phase1摘要/class/fmri_p1_data/normal/*"))
s2=sorted(glob.glob("C:/Users/user/Desktop/PPT/ICBHI/Phase1摘要/class/fmri_p1_data/sad/*"))
X_path_ls = s0+s1+s2
y_ls=[]

for i in X_path_ls:
    c=i.split("/")[-1].split("\\")[0]
    # c=i.split("\\")[-2]
    y_ls.append(c)
encoder = LabelBinarizer()
y_ls = encoder.fit_transform(y_ls) 

x =  X_path_ls
y =  y_ls
x_train, x_val, y_train, y_val = train_test_split(x, y,train_size=0.8,random_state=410,stratify=y)
x_val, x_test, y_val, y_test = train_test_split(x_val, y_val,train_size=0.5,random_state=410,stratify=y_val)

test_name_list=[]
for i in range(len(x_test)):
    test_name_list_=x_test[i].split("\\")[-1].split(".")[0]
    test_name_list.append(test_name_list_)

train_distribution=pd.Series(np.argmax(y_train,axis=-1)).value_counts(normalize=True).tolist()
val_distribution=pd.Series(np.argmax(y_val,axis=-1)).value_counts(normalize=True).tolist()
test_distribution=pd.Series(np.argmax(y_test,axis=-1)).value_counts(normalize=True).tolist()

y_train.astype(np.float64)
y_val.astype(np.float64)
y_test.astype(np.float64)


count = 0
fs = 400
npy_data_batch = []
for i in x_train:
    npy_data = load_npz_data(i)
    npy_data = (npy_data-np.min(npy_data))/(np.max(npy_data)-np.min(npy_data)) 
    npy_data = npy_data.T
    
    # for k in range(len(npy_data[0])):
    #     G = npy_data[:,k]
    #     derivative1 = np.gradient(G)
    #     npy_data[:,k] = (derivative1-np.min(derivative1))/(np.max(derivative1)-np.min(derivative1)) 
    npy_data_batch.append(npy_data)

x_train = np.array(npy_data_batch)
x_train = np.expand_dims(x_train, axis=-1)

npy_data_batch = []
for i in x_val:
    npy_data = load_npz_data(i)
    npy_data = (npy_data-np.min(npy_data))/(np.max(npy_data)-np.min(npy_data)) 
    npy_data = npy_data.T
    # for k in range(len(npy_data[0])):
    #     G = npy_data[:,k]
    #     derivative1 = np.gradient(G)
    #     npy_data[:,k] = (derivative1-np.min(derivative1))/(np.max(derivative1)-np.min(derivative1)) 
    npy_data_batch.append(npy_data)
x_val = np.array(npy_data_batch)
x_val = np.expand_dims(x_val, axis=-1)

npy_data_batch = []
for i in x_test:
    npy_data = load_npz_data(i)
    npy_data = (npy_data-np.min(npy_data))/(np.max(npy_data)-np.min(npy_data)) 
    npy_data = npy_data.T
    # for k in range(len(npy_data[0])):
    #     G = npy_data[:,k]
    #     derivative1 = np.gradient(G)
    #     npy_data[:,k] = (derivative1-np.min(derivative1))/(np.max(derivative1)-np.min(derivative1)) 
    npy_data_batch.append(npy_data)
x_test = np.array(npy_data_batch)
x_test = np.expand_dims(x_test, axis=-1)


inputA = Input(shape=(25,246))
inputB = Input(shape=(25,246))
inputC = Input(shape=(25,246))

F = SE_Block(inputA)
F_shortcut1 = F

F = Conv1D(filters=32, kernel_size=(3), strides=1,padding='same', activation='relu',kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.01))(F)
F = SE_Block(F)
F = BatchNormalization()(F)
# F = AveragePooling1D(2)(F)

F_shortcut2 = F

F = Conv1D(filters=64, kernel_size=(3),strides=1, padding='same', activation='relu',kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.01))(F)
F = SE_Block(F)
F = BatchNormalization()(F)
# F = AveragePooling1D(2,)(F)

# Shortcut path
F_shortcut_A = Conv1D(filters=64, kernel_size=(3), strides=1, padding='same')(F_shortcut1)
F_shortcut_A = SE_Block(F_shortcut_A)
F_shortcut_A = BatchNormalization()(F_shortcut_A)
F = Add()([F,F_shortcut_A])



F = Conv1D(filters=128, kernel_size=(3), strides=1,padding='same', activation='relu',kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.01))(F)
F = SE_Block(F)
F = BatchNormalization()(F)
# F = AveragePooling1D(2)(F)

# Shortcut path
F_shortcut2 = Conv1D(filters=128, kernel_size=(3), strides=1, padding='same')(F_shortcut2)
F_shortcut2 = SE_Block(F_shortcut2)
F_shortcut2 = BatchNormalization()(F_shortcut2)

F_shortcut_B = Conv1D(filters=128, kernel_size=(3), strides=1, padding='same')(F_shortcut1)
F_shortcut_B = SE_Block(F_shortcut_B)
F_shortcut_B = BatchNormalization()(F_shortcut_B)

F = Add()([F,F_shortcut_B,F_shortcut2])

F = Flatten()(F)
F = Dense(512, activation="relu",kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.01))(F)
F = Dense(100, activation="relu",kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.01))(F)
F = Dense(3, activation="softmax")(F)
model = Model(inputA, F)
model.summary()

# import tensorflow_addons as tfa
radam = tfa.optimizers.RectifiedAdam(0.001)
ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)


optimizer = ranger
model.compile(optimizer=ranger,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# model.compile(optimizer='sgd', loss=tf.keras.losses.KLDivergence(),metrics=['accuracy'])

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath="./models/best.h5",
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
# Step 6: 訓練模型
#%%
history  = model.fit(x_train, y_train,validation_data=(x_val,y_val), epochs=10000, batch_size=128,callbacks=[model_checkpoint_callback])
hist_df = pd.DataFrame(history.history) 
hist_csv_file = './models/history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
#%%
fig1 = plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# plt.show()
plt.savefig('./models/class_loss.png')
fig2 = plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
# plt.show()
plt.savefig('./models/class_acc.png')
# Step 7: 評估模型（如果有測試數據的話）
# test_loss, test_accuracy = model.evaluate(x_test, y_test)
# print("Test Loss:", test_loss)
# print("Test Accuracy:", test_accuracy)
#%%inside_test
model.load_weights("./models/best.h5")
results = model.evaluate(x_test, y_test)
pred_list=[]
happy_confidence=[]
normal_confidence=[]
sad_confidence=[]
for p in range(len(x_test)):
    pred = model(np.expand_dims(x_test[p],axis=0))
    happy_confidence_ = pred.numpy()[0,0]
    normal_confidence_ = pred.numpy()[0,1]
    sad_confidence_ = pred.numpy()[0,2]
    
    happy_confidence.append(happy_confidence_)
    normal_confidence.append(normal_confidence_)
    sad_confidence.append(sad_confidence_)
    if np.argmax(pred.numpy())==0 :
        pred=1
        
    elif np.argmax(pred.numpy())==1 :
        pred=0

    elif np.argmax(pred.numpy())==2 :
        pred=-1
    
    pred_list.append(pred)
y_test_ =np.argmax(y_test,axis=1).tolist()
for p in range(len(y_test_)):
    if y_test_[p]==0 :
        y_test_[p]=1
        
    elif y_test_[p]==1 :
        y_test_[p]=0

    elif y_test_[p]==2 :
        y_test_[p]=-1

log=("./models/CLASS_inner_test.csv")
d1 = {'test_name_list':test_name_list,'GT':y_test_,'PRE':pred_list,'happy_confidence':happy_confidence,'normal_confidence':normal_confidence,'sad_confidence':sad_confidence}
df = pd.DataFrame(d1)
df.to_csv(log,index=0)

# 将列表转换为 numpy 数组
y_true = np.array(y_test_)
y_pred = np.array(pred_list)


from sklearn.metrics import classification_report
print(classification_report(y_test_, pred_list))

#%%outside_test
counter = 0
model.load_weights("./models/best.h5")
label_list=[]
pred_list=[]
Participant=[]
Trial=[]
happy_confidence=[]
normal_confidence=[]
sad_confidence=[]
tX_path_ls = sorted(glob.glob("../test/fmri/*"))
tX_path_ls = natsorted(tX_path_ls)
for p in tX_path_ls:
    Trial_ = tX_path_ls[counter].split("_")[-1].split(".")[0]
    Participant_ = tX_path_ls[counter].split("\\")[-1].split("t")[0]
    npy_data = load_npz_data(p)
    npy_data = (npy_data-np.min(npy_data))/(np.max(npy_data)-np.min(npy_data)) 
    npy_data = npy_data.T
    npy_data = np.expand_dims(npy_data, axis=0)
    pred = model(npy_data)
    happy_confidence_ = pred.numpy()[0,0]
    normal_confidence_ = pred.numpy()[0,1]
    sad_confidence_ = pred.numpy()[0,2]
    
    happy_confidence.append(happy_confidence_)
    normal_confidence.append(normal_confidence_)
    sad_confidence.append(sad_confidence_)
    Participant.append(Participant_)
    Trial.append(Trial_)
    pred_list.append(np.argmax(pred.numpy()))
    counter+=1

for i in range(len(pred_list)):
    if pred_list[i]==0:
        pred_list[i]=1
        continue
    elif pred_list[i]==1:
        pred_list[i]=0
        continue
    elif pred_list[i]==2:
        pred_list[i]=-1
        continue
log=("./models/outer_CLASS.csv")
d1 = {'Participant':Participant,'Trial':Trial,'CLASS': pred_list,'happy_confidence':happy_confidence,'normal_confidence':normal_confidence,'sad_confidence':sad_confidence}
df = pd.DataFrame(d1)
df.to_csv(log,index=0)

#%%存訓練測驗測試資料檔名 (暫時不用管)
# def pad_dict_list(dict_list, padel):
#     lmax = 0
#     for lname in dict_list.keys():
#         lmax = max(lmax, len(dict_list[lname]))
#     for lname in dict_list.keys():
#         ll = len(dict_list[lname])
#         if  ll < lmax:
#             dict_list[lname] += [padel] * (lmax - ll)
#     return dict_list
# log=("./models/classsplit.csv")
# d1 = {'train_list':x_train,'val_list':x_val,'test_list':x_test,"train_distribution":train_distribution,"val_distribution":train_distribution,"test_distribution":train_distribution}

# d1 = pad_dict_list(d1, 0)
# df = pd.DataFrame(d1)
# df.to_csv(log,index=0)

#%%

