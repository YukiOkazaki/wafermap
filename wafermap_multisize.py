#!/usr/bin/env python
# coding: utf-8

# ## ウエハサイズを限定せずに機械学習させる

# ### import，入力データの読み込み

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# import numpy as np # linear algebra
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


import os
from os.path import join
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import csv

import pickle
import copy
import cv2

from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras
from tensorflow.keras import layers, Input, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier 
import keras.backend.tensorflow_backend as tfback


import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

datapath = join('data', 'wafer')

print(os.listdir("../input"))
import warnings
warnings.filterwarnings("ignore")

from memory_profiler import profile

MAKE_DATASET = False


# ### データについて

# In[3]:


if MAKE_DATASET:
    df=pd.read_pickle("../input/LSWMD.pkl")

    df = df.drop(['waferIndex'], axis = 1)

    def find_dim(x):
        dim0=np.size(x,axis=0)
        dim1=np.size(x,axis=1)
        return dim0,dim1
    df['waferMapDim']=df.waferMap.apply(find_dim)


# In[4]:


if MAKE_DATASET:
    df['failureNum']=df.failureType
    df['trainTestNum']=df.trianTestLabel
    mapping_type={'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,'Random':5,'Scratch':6,'Near-full':7,'none':8}
    mapping_traintest={'Training':0,'Test':1}
    df=df.replace({'failureNum':mapping_type, 'trainTestNum':mapping_traintest})

    tol_wafers = df.shape[0]

    df_withlabel = df[(df['failureNum']>=0) & (df['failureNum']<=8)]
    df_withlabel =df_withlabel.reset_index()
    df_withpattern = df[(df['failureNum']>=0) & (df['failureNum']<=7)]
    df_withpattern = df_withpattern.reset_index()
    df_nonpattern = df[(df['failureNum']==8)]


# ### データサイズ関係なく処理

# - 使えるデータサイズを求める
#     - None以外の合計が50個以上のウエハ
#     - サイズが100以下
#     - 統一サイズを100

# In[5]:


if MAKE_DATASET:
    uni_waferDim=np.unique(df.waferMapDim, return_counts=True)
    wdim = uni_waferDim[0]
    failure_list = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full', 'none']
    usable_wdim_list = []
    usable_wafer_num = 0
    max_size = 100
    for i in range(len(wdim)):
        sub_df = df.loc[df['waferMapDim'] == wdim[i]]
        pattern_num = [0] * 9
        for j in range(len(sub_df)):
            if len(sub_df.iloc[j,:]['failureType']) == 0:
                continue
            pattern = sub_df.iloc[j,:]['failureType'][0][0]
            pattern_num[failure_list.index(pattern)] += 1
        if sum(pattern_num) - pattern_num[8] >= 50 and wdim[i][0] <= max_size and wdim[i][1] <= max_size:
            usable_wdim_list.append(wdim[i])
            print(wdim[i], len(sub_df), sum(pattern_num))
            usable_wafer_num += sum(pattern_num)
    print(usable_wafer_num)


# In[6]:


def make_unisize_wafer(size, wafer):
    width, height = wafer.shape
    unisize_wafer = np.zeros((size, size))
    width_pad = int((size - width) / 2)
    height_pad = int((size - height) / 2)
    unisize_wafer[width_pad:width_pad + width, height_pad:height_pad + height] = wafer
    return unisize_wafer


# In[7]:


if MAKE_DATASET:
    sw = np.ones((usable_wafer_num, max_size, max_size), dtype='int8')
    label = list()
    count = 0
    for usable_wdim in usable_wdim_list:
        sub_df = df.loc[df['waferMapDim'] == usable_wdim]
        sub_wafer = sub_df['waferMap'].values
        print(usable_wdim)
        print(len(sub_df))

        for i in range(len(sub_df)):
            # skip null label
            if len(sub_df.iloc[i,:]['failureType']) == 0:
                continue
            sw[count] = make_unisize_wafer(max_size, sub_df.iloc[i,:]['waferMap'])
            label.append(sub_df.iloc[i,:]['failureType'][0][0])
            count += 1
            if i % 1000 == 0:
                print(" ", i)
    x = sw
    y = np.array(label).reshape((-1,1))


# ### xとyをファイルに保存

# In[8]:


if MAKE_DATASET:
    faulty_case = np.unique(y)
    print('Faulty case list : {}'.format(faulty_case))
if not MAKE_DATASET:
    faulty_case = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'none']


# In[9]:


if MAKE_DATASET:
    for f in faulty_case :
        print('{} : {}'.format(f, len(y[y==f])))


# In[10]:


if MAKE_DATASET:
    for i, l in enumerate(faulty_case):
        y[y==l] = int(i)
        print(type(i))
    y = y.astype(np.int8)


# In[11]:


from sklearn.externals import joblib

def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj,f,protocol=4)

def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data

if MAKE_DATASET:
#     joblib.dump(x, './data/xmulti.pickle')
    joblib.dump(y, './data/ymulti.pickle')
    
if not MAKE_DATASET:
#     x = joblib.load('./data/xmulti.pickle')
    y = joblib.load('./data/ymulti.pickle')


# In[12]:


if MAKE_DATASET:
    print('x shape : {}, y shape : {}'.format(x.shape, y.shape))


# In[13]:


for i in range(9) :
    print('{} : {}'.format(i, len(y[y==i])))


# - 最初のデータを可視化してみる．

# In[14]:


if MAKE_DATASET:
    # plot 1st data
    plt.imshow(x[0,:, :, 0])
    plt.show()

    # check faulty case
    print('Faulty case : {} '.format(faulty_case[y[0]]))


# In[15]:


if MAKE_DATASET:
    x = x.reshape((-1, 100, 100, 1))
    x.shape


# - 14366枚の26x26ウエハの不良パターンは上記のようになっている．

# In[16]:


if MAKE_DATASET:
    new_x = np.zeros((len(x), 100, 100, 3), dtype='int8')

    for w in range(len(x)):
        for i in range(100):
            for j in range(100):
                new_x[w, i, j, int(x[w, i, j])] = 1
        print(w)


# In[17]:


if MAKE_DATASET:
    joblib.dump(new_x, './data/new_xmulti.pickle')
    
if not MAKE_DATASET:
    new_x = joblib.load('./data/new_xmulti.pickle')


# In[18]:


# # cupyに変換
# print(type(new_x))
# print(type(y))
# new_x = np.asarray(new_x)
# y = np.asarray(y)
# print(type(new_x))
# print(type(y))


# - new_xを(14366, 26, 26, 3)とし，最後の次元にはウエハの値(0, 1, 2)がそれぞれの値毎にベクトルとしてまとめられている．

# In[19]:


import sys

print("{}{: >25}{}{: >10}{}".format('|','Variable Name','|','Memory','|'))
print(" ------------------------------------ ")
for var_name in dir():
    if not var_name.startswith("_"):
        print("{}{: >25}{}{: >10}{}".format('|',var_name,'|',sys.getsizeof(eval(var_name)),'|'))


# ### オートエンコーダで学習

# #### エンコーダとデコーダのモデルを学習

# - モデルの定義をする．

# In[20]:


strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"], cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    # Encoder
    input_shape = (100, 100, 3)
    input_tensor = Input(input_shape)
    encode = layers.Conv2D(64, (3,3), padding='same', activation='relu')(input_tensor)

    latent_vector = layers.MaxPool2D()(encode)

    # Decoder
    decode_layer_1 = layers.Conv2DTranspose(64, (3,3), padding='same', activation='relu')
    decode_layer_2 = layers.UpSampling2D()
    output_tensor = layers.Conv2DTranspose(3, (3,3), padding='same', activation='sigmoid')

    # connect decoder layers
    decode = decode_layer_1(latent_vector)
    decode = decode_layer_2(decode)

    ae = models.Model(input_tensor, output_tensor(decode))
    ae.compile(optimizer = 'Adam',
                  loss = 'mse',
                 )


# In[21]:


ae.summary()


# - 層は
#     - 入力層
#     - 畳み込み層
#     - プーリング層
#     - 転置畳み込み層
#     - アップサンプリング層

# In[22]:


epoch=5
batch_size=128


# - 学習を開始する．
# - `new_x`を`new_x`にエンコードしデコードする．

# In[23]:


# start train
ae.fit(new_x, new_x,
       batch_size=batch_size,
       epochs=epoch,
       verbose=1)


# - エンコーダだけのモデルを定義する．

# In[24]:


encoder = models.Model(input_tensor, latent_vector)


# - デコーダだけのモデルを定義する．

# In[25]:


decoder_input = Input((50, 50, 64))
decode = decode_layer_1(decoder_input)
decode = decode_layer_2(decode)

decoder = models.Model(decoder_input, output_tensor(decode))


# - `encoder`を使って元のウエハ画像をエンコードする．

# In[26]:


# Encode original faulty wafer
# encoded_x = np.zeros((156581, 50, 50, 64), dtype="int8")
# encoded_x = (encoder.predict(new_x))


# - エンコードされた潜伏的な不良ウエハにノイズを負荷する．

# In[27]:


# Add noise to encoded latent faulty wafers vector.
# noised_encoded_x = encoded_x + np.random.normal(loc=0, scale=0.1, size = (len(encoded_x), 50, 50, 64))


# - 元のウエハ画像

# In[28]:


# check original faulty wafer data
# plt.imshow(np.argmax(new_x[3], axis=2))


# - ノイズが付加されたウエハ画像

# In[29]:


# # check new noised faulty wafer data
# noised_gen_x = np.argmax(decoder.predict(noised_encoded_x), axis=3)
# plt.imshow(noised_gen_x[3])


# ### データオーギュメンテーション

# - データオーギュメンテーションを行う関数を定義する．

# In[30]:


# augment function define
@profile
def gen_data(wafer, label):
    print(label)
    # Encode input wafer
    encoded_x = encoder.predict(wafer)
    
    # dummy array for collecting noised wafer
    gen_x = np.zeros((1, 100, 100, 3), dtype='int8')
    print(gen_x.shape)
    
    # Make wafer until total # of wafer to 2000
    for i in range((10000//len(encoded_x)) + 1):
        print(i)
        noised_encoded_x = encoded_x + np.random.normal(loc=0, scale=0.1, size = (len(encoded_x), 50, 50, 64)) 
        noised_gen_x = decoder.predict(noised_encoded_x)
        gen_x = np.concatenate((gen_x, noised_gen_x), axis=0)
    # also make label vector with same length
    gen_y = np.full((len(gen_x), 1), label)
    
    print(label, gen_x.shape)
    
    del encoded_x
    del noised_encoded_x
    del noised_gen_x
    
    # return date without 1st dummy data.
    return gen_x[1:], gen_y[1:]


# - 不良ラベルが付いているデータに対してデータオーギュメンテーションを行う．

# In[31]:


none_idx = np.where(y==8)[0][np.random.choice(len(np.where(y==8)[0]), size=117000, replace=False)]
new_x = np.delete(new_x, none_idx, axis=0)
y = np.delete(y, none_idx, axis=0)


# In[ ]:


# Augmentation for all faulty case.
for i in range(9) : 
    # skip none case
    if i == 8 : 
        continue
    
    gen_x, gen_y = gen_data(new_x[np.where(y==i)[0]], i)
    print("gen")
    new_x = np.concatenate((new_x, gen_x), axis=0)
    print("x")
    y = np.concatenate((y, gen_y))
    del gen_x
    del gen_y


# In[ ]:


print('After Generate new_x shape : {}, new_y shape : {}'.format(new_x.shape, y.shape))


# In[ ]:


for i in range(9) :
    print('{} : {}'.format(i, len(y[y==i])))


# In[ ]:


new_y = y


# - データオーギュメンテーションを行った結果，各不良データごとに2000枚増えた．
# - 合計は30707枚となった．

# - 不良ラベルのないデータは削除し，枚数を不良ラベルと同程度にする．

# In[ ]:


none_idx = np.where(y==8)[0][np.random.choice(len(np.where(y==8)[0]), size=120000, replace=False)]


# In[ ]:


new_x = np.delete(new_x, none_idx, axis=0)
new_y = np.delete(y, none_idx, axis=0)


# In[ ]:


print('After Delete "none" class new_x shape : {}, new_y shape : {}'.format(new_x.shape, new_y.shape))


# In[ ]:


for i in range(9) :
    print('{} : {}'.format(i, len(new_y[new_y==i])))


# - 削除した結果，全体は19707枚となった．

# ### 学習を行う
# - 不良ラベルを0-8の9次元のベクトルとして表現する．
# - one-hotエンコーディングを行っている．

# In[ ]:


# new_y = y


# In[ ]:


# for i, l in enumerate(faulty_case):
#     new_y[new_y==l] = i


# In[ ]:


# one-hot-encoding
new_y = to_categorical(new_y)


# - 学習データ（学習データと学習時のテストデータ）と最終的なテストデータに分割する．

# In[ ]:


# new_X=new_x[0:19000]
# new_Y=new_y[0:19000]
# test_x=new_x[19001:19706]
# test_y=new_y[19001:19706]
# test_x.shape
new_X = new_x
new_Y = new_y


# - 学習データを学習データと学習時のテストデータに分割する．

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(new_X, new_Y,
                                                    test_size=0.1,
                                                    random_state=2019)


# In[ ]:


print('Train x : {}, y : {}'.format(x_train.shape, y_train.shape))
print('Test x: {}, y : {}'.format(x_test.shape, y_test.shape))


# - 学習データ12730枚，テストデータ6270枚．

# - モデルの定義を行う．

# In[ ]:


def create_model():
    with tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"], 
                                        cross_device_ops = tf.distribute.HierarchicalCopyAllReduce()).scope():
        input_shape = (100, 100, 3)
        input_tensor = Input(input_shape)

        conv_1 = layers.Conv2D(16, (3,3), activation='relu', padding='same')(input_tensor)
        conv_2 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(conv_1)
        conv_3 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(conv_2)

        flat = layers.Flatten()(conv_3)

        dense_1 = layers.Dense(256, activation='relu')(flat)
        dense_2 = layers.Dense(64, activation='relu')(dense_1)
        output_tensor = layers.Dense(9, activation='softmax')(dense_2)

        model = models.Model(input_tensor, output_tensor)
        model.compile(optimizer='Adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

    return model


# - 3-Fold Cross validationで分割して学習する．

# In[ ]:


model = KerasClassifier(build_fn=create_model, epochs=30, batch_size=1, verbose=1) 
# 3-Fold Crossvalidation
kfold = KFold(n_splits=3, shuffle=True, random_state=2019) 
#results = cross_val_score(model, x_train, y_train, cv=kfold)
# Check 3-fold model's mean accuracy
#print('Simple CNN Cross validation score : {:.4f}'.format(np.mean(results)))


# - Cross validiationによる精度は99.10%であった．

# - Cross validationなしで学習する．

# In[ ]:


# del new_x
# del new_X


# In[ ]:


epoch=30
batch_size=256
model = create_model()


# In[ ]:


history = model.fit(x_train, y_train,
         validation_data=(x_test, y_test),
         epochs=epoch,
         batch_size=batch_size,
         verbose=1           
         )


# - テストデータで評価．    

# In[ ]:


score = model.evaluate(x_test, y_test)
#print('Test Loss:', score[0])
#print('Test accuracy:', score[1])
print('Testing Accuracy:',score[1])


# - acuurayは99.31%であった．

# - モデルは以下．
#     - 入力層
#     - 畳み込み層3つ
#     - Flatten層（1次元に）
#     - 全結合層3つ

# In[ ]:


model.summary()


# - accuracyグラフ，lossグラフは以下．
# - 5epoch程度で落ち着いている．

# In[ ]:


# accuracy plot 
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

