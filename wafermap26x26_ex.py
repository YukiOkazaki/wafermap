#!/usr/bin/env python
# coding: utf-8

# ## 26x26のウエハに限定して機械学習させる
# - データオーギュメンテーション（鏡映，回転を追加）

# ### import，入力データの読み込み

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
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

import numpy as np
import pandas as pd
import cv2

from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras
from keras import layers, Input, models
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier 


import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import random
random.seed(1)

datapath = join('data', 'wafer')

print(os.listdir("../input"))
import warnings
warnings.filterwarnings("ignore")


# ### データについて

# In[3]:


df=pd.read_pickle("../input/LSWMD.pkl")
df.info()


# - データセットは811,457枚のウエハマップから構成されている．

# - ウエハマップのそれぞれの列から得られる情報はないが，インスタンスごとにダイサイズが異なることがわかる．
# - ウエハマップのダイサイズをチェックするための変数`WaferMapDim`を定義する．（縦，横の値）

# In[4]:


df = df.drop(['waferIndex'], axis = 1)


# In[5]:


def find_dim(x):
    dim0=np.size(x,axis=0)
    dim1=np.size(x,axis=1)
    return dim0,dim1
df['waferMapDim']=df.waferMap.apply(find_dim)
df.sample(5)


# - 不良パターンと学習orテストラベルを数値で表す．

# In[6]:


df['failureNum']=df.failureType
df['trainTestNum']=df.trianTestLabel
mapping_type={'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,'Random':5,'Scratch':6,'Near-full':7,'none':8}
mapping_traintest={'Training':0,'Test':1}
df=df.replace({'failureNum':mapping_type, 'trainTestNum':mapping_traintest})


# In[7]:


tol_wafers = df.shape[0]
tol_wafers


# In[8]:


df_withlabel = df[(df['failureNum']>=0) & (df['failureNum']<=8)]
df_withlabel =df_withlabel.reset_index()
df_withpattern = df[(df['failureNum']>=0) & (df['failureNum']<=7)]
df_withpattern = df_withpattern.reset_index()
df_nonpattern = df[(df['failureNum']==8)]
df_withlabel.shape[0], df_withpattern.shape[0], df_nonpattern.shape[0]


# ### 26x26のデータに対して処理

# In[9]:


sub_df = df.loc[df['waferMapDim'] == (26, 26)]
sub_wafer = sub_df['waferMap'].values

sw = np.ones((1, 26, 26))
label = list()

for i in range(len(sub_df)):
    # skip null label
    if len(sub_df.iloc[i,:]['failureType']) == 0:
        continue
    sw = np.concatenate((sw, sub_df.iloc[i,:]['waferMap'].reshape(1, 26, 26)))
    label.append(sub_df.iloc[i,:]['failureType'][0][0])


# In[10]:


x = sw[1:]
y = np.array(label).reshape((-1,1))


# In[11]:


mask_x = np.zeros((24, 24))
dummy_x = cv2.resize(x[0], (24,24))
mask_x[dummy_x == 1] = 1 
mask_x[dummy_x == 2] = 1 
mask_x = mask_x.reshape((1, 24,24))


# In[12]:


print('x shape : {}, y shape : {}'.format(x.shape, y.shape))


# - 26x26のウエハが14366枚抽出できた．

# - 最初のデータを可視化してみる．
# - その前に，26x26のデータでおかしなものを表示

# In[13]:


for i in range(len(x)):
    error = np.where((x[0] != x[i]) & ((x[0] == 0) | (x[i] == 0)))
    if len(error[0]) > 0:
        print(str(i) + " is error")
        #print(error)


# In[14]:


# plot 1st data
plt.imshow(x[0])
plt.show()

# check faulty case
print('Faulty case : {} '.format(y[0]))


# - おかしなウエハは除去する

# In[15]:


error_list = []
for i in range(len(x)):
    error = np.where((x[0] != x[i]) & ((x[0] == 0) | (x[i] == 0)))
    if len(error[0]) > 0:
        error_list.append(i)
x = np.delete(x, error_list, 0)
y = np.delete(y, error_list, 0)
print(x.shape)
print(y.shape)


# - 形が異なるウエハを削除したところ，14350枚となった．

# In[16]:


x = x.reshape((-1, 26, 26, 1))
x.shape


# In[17]:


faulty_case = np.unique(y)
print('Faulty case list : {}'.format(faulty_case))


# In[18]:


faulty_case_dict = dict()


# In[19]:


for i, f in enumerate(faulty_case) :
    print('{} : {}'.format(f, len(y[y==f])))
    faulty_case_dict[i] = f


# - 14366枚の26x26ウエハの不良パターンは上記のようになっている．

# In[20]:


new_x = np.zeros((len(x), 26, 26, 3))

for w in range(len(x)):
    for i in range(26):
        for j in range(26):
            new_x[w, i, j, int(x[w, i, j])] = 1


# In[21]:


new_x.shape


# - new_xを(14366, 26, 26, 3)とし，最後の次元にはウエハの値(0, 1, 2)がそれぞれの値毎にベクトルとしてまとめられている．
# - ウエハデータの各ピクセルは，0:ウエハなし，1:正常，2:不良を表す．

# ### テストデータに分割
# - ランダムなリストを生成する

# In[22]:


def rand_ints_nodup(a, b, k):
    ns = []
    while len(ns) < k:
        n = random.randint(a, b)
        if not n in ns:
            ns.append(n)
    return ns


# In[23]:


testsize = 500
randlist = rand_ints_nodup(0, new_x.shape[0]-1, testsize)


# - tempx, tempyにテストデータを分割
# - new_x, yからその分を削除

# In[24]:


tempx = new_x.copy()[randlist, :, :, :]
tempy = y.copy()[randlist, :]


# In[25]:


for f in faulty_case :
    print('{} : {}'.format(f, len(tempy[tempy==f])))


# In[26]:


new_x = np.delete(new_x, randlist, axis=0)
y = np.delete(y, randlist, axis=0)


# - バリデーションデータを生成

# - valx, valyにテストデータを分割
# - new_x, yからその分を削除

# In[27]:


testsize = 1000
randlist = rand_ints_nodup(0, new_x.shape[0]-1, testsize)

valx = new_x.copy()[randlist, :, :, :]
valy = y.copy()[randlist, :]

for f in faulty_case :
    print('{} : {}'.format(f, len(tempy[tempy==f])))
    
new_x = np.delete(new_x, randlist, axis=0)
y = np.delete(y, randlist, axis=0)


# ### オートエンコーダで学習

# #### エンコーダとデコーダのモデルを学習

# - モデルの定義をする．

# In[28]:


# Encoder
input_shape = (26, 26, 3)
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


# In[29]:


ae.summary()


# - 層は
#     - 入力層
#     - 畳み込み層
#     - プーリング層
#     - 転置畳み込み層
#     - アップサンプリング層

# In[30]:


epoch=30
batch_size=1024


# - 学習を開始する．
# - `new_x`を`new_x`にエンコードしデコードする．

# In[31]:


# start train
ae.fit(new_x, new_x,
       batch_size=batch_size,
       epochs=epoch,
       verbose=1)


# - エンコーダだけのモデルを定義する．

# In[32]:


encoder = models.Model(input_tensor, latent_vector)


# - デコーダだけのモデルを定義する．

# In[33]:


decoder_input = Input((13, 13, 64))
decode = decode_layer_1(decoder_input)
decode = decode_layer_2(decode)

decoder = models.Model(decoder_input, output_tensor(decode))


# - `encoder`を使って元のウエハ画像をエンコードする．

# In[34]:


# Encode original faulty wafer
encoded_x = encoder.predict(new_x)


# - エンコードされた潜伏的な不良ウエハにノイズを負荷する．

# In[35]:


# Add noise to encoded latent faulty wafers vector.
noised_encoded_x = encoded_x + np.random.normal(loc=0, scale=0.1, size = (len(encoded_x), 13, 13, 64))


# - 元のウエハ画像

# In[36]:


# check original faulty wafer data
plt.imshow(np.argmax(new_x[3], axis=2))


# - マスクの定義

# In[37]:


# 0がウエハ領域，1が範囲外
mask = new_x[0, :, :, 0].copy()
#out_region = np.where(mask == 1.0)
#in_region = np.where(mask == 0.0)
#mask[out_region] = 0.0
#mask[in_region] = 1.0
#print(mask)


# - 回転の実験
# 
# 90度ごとなら自作OpenCV関数が有用

# - 90度以外の回転

# In[38]:


from PIL import Image, ImageOps
def rotation_pil_mask(img, degree):
    src = Image.fromarray(np.uint8(img))
    img_rotate = np.array(src.rotate(degree))
    
    #before
    '''
    plt.imshow(np.argmax(img_rotate, axis=2))
    plt.title("rotate " + str(degree) + " degree before")
    plt.show()
    '''
    
    #マスクの適用, 
    out_region = np.where((np.argmax(img_rotate, axis=2) != 0) & (mask > 0.0))
    img_rotate[out_region[0], out_region[1]] = np.array([1., 0., 0.])
    in_region = np.where((np.argmax(img_rotate, axis=2) == 0) & (mask == 0.0))
    img_rotate[in_region[0], in_region[1]] = np.array([0., 1., 0.])

    #after
    '''plt.imshow(np.argmax(img_rotate, axis=2))
    plt.title("rotate " + str(degree) + " degree after")
    plt.show()'''

    return img_rotate

# 上下方向の鏡映
def flip_pil_mask(img):
    src = Image.fromarray(np.uint8(img))
    img_flip = np.array(ImageOps.flip(src))
    
    #マスクの適用, 
    out_region = np.where((np.argmax(img_flip, axis=2) != 0) & (mask > 0.0))
    img_flip[out_region[0], out_region[1]] = np.array([1., 0., 0.])
    in_region = np.where((np.argmax(img_flip, axis=2) == 0) & (mask == 0.0))
    img_flip[in_region[0], in_region[1]] = np.array([0., 1., 0.])
    
    return img_flip

# 左右方向の鏡映
def mirror_pil_mask(img):
    src = Image.fromarray(np.uint8(img))
    img_mirror = np.array(ImageOps.mirror(src))
    
    #マスクの適用, 
    out_region = np.where((np.argmax(img_mirror, axis=2) != 0) & (mask > 0.0))
    img_mirror[out_region[0], out_region[1]] = np.array([1., 0., 0.])
    in_region = np.where((np.argmax(img_mirror, axis=2) == 0) & (mask == 0.0))
    img_mirror[in_region[0], in_region[1]] = np.array([0., 1., 0.])
    
    return img_mirror


# In[39]:


'''
wafer = new_x[np.where(y=="Edge-Loc")[0]].reshape(len(np.where(y=="Edge-Loc")[0]), 26, 26, 3)
#plt.imshow(np.argmax(wafer[0], axis=2))
#plt.show()

print(wafer[0].shape)

for i in range(19):
    rotation_pil_mask(wafer[0], i*20)
'''


# In[ ]:





# - ノイズが付加されたウエハ画像

# In[40]:


# check new noised faulty wafer data
noised_gen_x = np.argmax(decoder.predict(noised_encoded_x), axis=3)
plt.imshow(noised_gen_x[3])


# ### データオーギュメンテーション

# - データオーギュメンテーションを行う関数を定義する．
# - 鏡映，回転を行う

# In[41]:


# augment function define (add rotate, flip)
def gen_data(wafer, label):
    
    # dummy array for collecting noised wafer
    gen_x = np.zeros((1, 26, 26, 3))
    aug_x = np.zeros((1, 26, 26, 3))
    
    flipflag = True
    noiseflag = True

    ite = 10 if label != 'none' else 1
    for i in range(len(wafer)):
        for j in range(ite):
#             rotatedata = wafer[i].reshape(1, 26, 26, 3)
            
            angle = int(180 / ite)
            if not flipflag:
                rotatedata = rotation_pil_mask(wafer[i], j * angle)
                rotatedata = rotatedata.reshape(1, 26, 26, 3)
                aug_x = np.concatenate((aug_x, rotatedata), axis = 0)
            else:
                rotatedata = rotation_pil_mask(wafer[i], j * angle)
                flipdata = flip_pil_mask(rotatedata)
                flipmirrordata = mirror_pil_mask(flipdata)
                mirrordata = mirror_pil_mask(rotatedata)
                rotatedata = rotatedata.reshape(1, 26, 26, 3)
                flipdata = flipdata.reshape(1, 26, 26, 3)
                flipmirrordata = flipmirrordata.reshape(1, 26, 26, 3)
                mirrordata = mirrordata.reshape(1, 26, 26, 3)
                aug_x = np.concatenate((aug_x, rotatedata, flipdata, flipmirrordata, mirrordata), axis = 0)
#             aug_x = np.concatenate((aug_x, rotatedata), axis = 0)
    
    aug_x = aug_x[1:]
    encoded_x = encoder.predict(aug_x)
    print(encoded_x.shape)
        
    
    # Make wafer until total # of wafer to 15000
#     if label != 'none':
#         for i in range((30000//len(encoded_x)) + 1):
#             noised_encoded_x = encoded_x + np.random.normal(loc=0, scale=0.1, size = (len(encoded_x), 13, 13, 64)) 
# #             noised_encoded_x = encoded_x
#             noised_gen_x = decoder.predict(noised_encoded_x)
#             gen_x = np.concatenate((gen_x, noised_gen_x), axis=0)
#     else:
#         gen_x = aug_x

    for i in range((30000//len(encoded_x)) + 1):
        if noiseflag:
            noised_encoded_x = encoded_x + np.random.normal(loc=0, scale=0.1, size = (len(encoded_x), 13, 13, 64)) 
        else :
            noised_encoded_x = encoded_x
        noised_gen_x = decoder.predict(noised_encoded_x)
        gen_x = np.concatenate((gen_x, noised_gen_x), axis=0)
        
    # also make label vector with same length
    gen_y = np.full((len(gen_x), 1), label)
    
    # return date without 1st dummy data.
    return gen_x[1:], gen_y[1:]


# - データオーギュメンテーション（ノイズ付加）したものが適切か調べる
# 
# 一番左が元画像
# 右がデータオーギュメンテーション後

# In[42]:


# backup
bu_new_x = new_x.copy()
bu_y = y.copy()


# In[43]:


# new_x = bu_new_x.copy()
# y = bu_y.copy()
# gen_x = []


# - 不良ラベルが付いているデータに対してデータオーギュメンテーションを行う．

# In[44]:


# Augmentation for all faulty case.
for f in faulty_case : 
    # skip none case
    if f == 'none' : 
        continue
    
    gen_x, gen_y = gen_data(new_x[np.where(y==f)[0]], f)
    new_x = np.concatenate((new_x, gen_x), axis=0)
    y = np.concatenate((y, gen_y))


# In[45]:


print('After Generate new_x shape : {}, new_y shape : {}'.format(new_x.shape, y.shape))


# In[46]:


for f in faulty_case :
    print('{} : {}'.format(f, len(y[y==f])))
new_y = y


# - ノイズ付加後におかしなデータがあるかチェック
# - 元データとずれがあるもの：1689枚

# ### ノイズ付加後もマスクで直す

# In[47]:


backup_new_x = new_x.copy()
backup_new_y = new_y.copy()


# In[48]:


count = 0
for i in range(len(new_x)):
    error = np.where((np.argmax(new_x[0], axis=2) != np.argmax(new_x[i], axis=2)) & (np.argmax(new_x[0], axis=2) == 0))
    if len(error[0]) > 0:
        #print(str(i) + "error")
        #print(error)
        count += 1
print(count)

for i in range(len(new_x)):
    #マスクの適用, 
    out_region = np.where((np.argmax(new_x[i], axis=2) != 0) & (mask > 0.0))
    new_x[i, out_region[0], out_region[1]] = np.array([1., 0., 0.])
    in_region = np.where((np.argmax(new_x[i], axis=2) == 0) & (mask == 0.0))
    new_x[i, in_region[0], in_region[1]] = np.array([0., 1., 0.])
    
count = 0
for i in range(len(new_x)):
    error = np.where((np.argmax(new_x[0], axis=2) != np.argmax(new_x[i], axis=2)) & (np.argmax(new_x[0], axis=2) == 0))
    if len(error[0]) > 0:
        #print(str(i) + "error")
        #print(error)
        count += 1
print(count)


'''plt.imshow(np.argmax(new_x[14397], axis=2))
plt.show()
plt.imshow(np.argmax(new_x[0], axis=2))
plt.show()'''


# - データオーギュメンテーションを行った結果，各不良データごとに約40000枚に増えた．
# - 合計は368112枚となった．

# In[49]:


# x = [0,1,2,3,4,5,6,7,8]
# labels2 = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full','none']

# for k in x:
#     fig, ax = plt.subplots(nrows = 1, ncols = 10, figsize=(20, 20))
#     ax = ax.ravel(order='C')
#     for j in [k]:
#         index = np.where(new_y==labels2[j])[0]
#         img = new_x[index]
#         #img = new_x[0:10]
#         for i in range(10):
#             ax[i].imshow(np.argmax(img[i+0], axis=2))
#             ax[i].set_title(new_y[index[i+0]], fontsize=15)
#             #ax[i].set_xlabel(df_withpattern.index[img.index[i]], fontsize=10)
#             ax[i].set_xticks([])
#             ax[i].set_yticks([])
#     plt.tight_layout()
#     plt.show() 


# ### 学習を行う
# - 不良ラベルを0-8の9次元のベクトルとして表現する．
# - one-hotエンコーディングを行っている．

# In[50]:


for i, l in enumerate(faulty_case):
    new_y[new_y==l] = i
    tempy[tempy==l] = i
    valy[valy==l] = i


# In[51]:


# one-hot-encoding
new_y = to_categorical(new_y)
tempy = to_categorical(tempy)
valy = to_categorical(valy)


# In[52]:


new_X=new_x
new_Y=new_y


# - 学習データとテストデータに分割する．

# In[53]:


x_train, x_test, y_train, y_test = train_test_split(new_X, new_Y,
                                                    test_size=0.33,
                                                    random_state=2019)

# x_train = new_X
# x_test = tempx
# y_train = new_Y
# y_test = tempy

x_train = new_X
x_test = valx
y_train = new_Y
y_test = valy


# In[54]:


print('Train x : {}, y : {}'.format(x_train.shape, y_train.shape))
print('Test x: {}, y : {}'.format(x_test.shape, y_test.shape))


# - 学習データ246635枚，テストデータ121477枚．

# - モデルの定義を行う．

# ### CNN

# In[55]:


input_shape = (26, 26, 3)
input_tensor = Input(input_shape)
def create_model():


    conv_1 = layers.Conv2D(16, (3,3), activation='relu', padding='same')(input_tensor)
    conv_2 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(conv_1)
    conv_3 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(conv_2)

    flat = layers.Flatten()(conv_3)

    dense_1 = layers.Dense(512, activation='relu')(flat)
    dense_2 = layers.Dense(128, activation='relu')(dense_1)
    output_tensor = layers.Dense(9, activation='softmax')(dense_2)

    model = models.Model(input_tensor, output_tensor)
    model.compile(optimizer='Adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

    return model


# - 3-Fold Cross validationで分割して学習する．

# In[56]:


model = KerasClassifier(build_fn=create_model, epochs=30, batch_size=1024, verbose=1) 
# 3-Fold Crossvalidation
kfold = KFold(n_splits=3, shuffle=True, random_state=2019) 
results = cross_val_score(model, x_train, y_train, cv=kfold)
# Check 3-fold model's mean accuracy
print('Simple CNN Cross validation score : {:.4f}'.format(np.mean(results)))


# - Cross validiationによる精度は99.55%であった．

# - Cross validationなしで学習する．

# In[57]:


history = model.fit(x_train, y_train,
         validation_data=[x_test, y_test],
         epochs=epoch,
         batch_size=batch_size,
         verbose=1           
         )


# - テストデータで評価．    

# In[58]:


score = model.score(x_test, y_test)
#print('Test Loss:', score[0])
#print('Test accuracy:', score[1])
print('Testing Accuracy:',score)
valiscore = score


# - データオーギュメンテーションなしの実データで評価

# In[59]:


score = model.score(tempx, tempy)
#print('Test Loss:', score[0])
#print('Test accuracy:', score[1])
print('Testing Accuracy:',score)
testscore = score


# - acuurayは99.70%であった．

# - モデルは以下．
#     - 入力層
#     - 畳み込み層3つ
#     - Flatten層（1次元に）
#     - 全結合層3つ

# In[60]:


model.model.summary()


# - accuracyグラフ，lossグラフは以下．
# - 5epoch程度で落ち着いている．

# In[61]:


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


# In[62]:


#y_train_pred = np.argmax(model.predict(x_train))
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
y_train_max = np.argmax(y_train, axis=1)
y_test_max = np.argmax(y_test, axis=1)

print(y_train_max[0])
print(y_train_pred[0])

train_acc2 = np.sum(y_train_max == y_train_pred, axis=0, dtype='float') / x_train.shape[0]
test_acc2 = np.sum(y_test_max == y_test_pred, axis=0, dtype='float') / x_test.shape[0]
print('Training acc: {}'.format(train_acc2*100))
print('Testing acc: {}'.format(test_acc2*100))
print("y_train_pred[:100]: ", y_train_pred[:100])
print ("y_train_max[:100]: ", y_train_max[:100])
trainscore = train_acc2


# ### 混同行列

# In[63]:


import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[64]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test_max, y_test_pred)
np.set_printoptions(precision=2)

from matplotlib import gridspec
fig = plt.figure(figsize=(8, 15)) 
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1]) 

## Plot non-normalized confusion matrix
plt.subplot(gs[0])
plot_confusion_matrix(cnf_matrix, title='Confusion matrix')

# Plot normalized confusion matrix
plt.subplot(gs[1])
plot_confusion_matrix(cnf_matrix, normalize=True, title='Normalized confusion matrix')

plt.show()


# ### クラス活性化マップ

# In[65]:


#set target wafer number
target_wafer_num = 100
# predict 
prob = model.model.predict(x_test[target_wafer_num].reshape(1,26,26,3))


# In[66]:


aver_output = model.model.layers[3]
aver_model = models.Model(input_tensor, aver_output.output)
cam_result = aver_model.predict(x_test[target_wafer_num].reshape(1, 26, 26, 3))


# In[67]:


print(model.model.layers[3].name)


# In[68]:


weight_result = model.model.layers[-1].get_weights()[0]


# In[69]:


cam_result.shape


# In[70]:


mask_x = np.zeros((26, 26))
dummy_x = cv2.resize(x[0], (26,26))
mask_x[dummy_x == 1] = 1 
mask_x[dummy_x == 2] = 1 
mask_x = mask_x.reshape((1, 26,26))


# In[71]:


def make_cam(cam_result, weight_result): 
    cam_arr = np.zeros((1,26, 26))
    for row in range(0,9):
        cam = np.zeros((1, 26, 26))
        for i, w in enumerate(weight_result[:, row]):
            cam += (w*cam_result[0,:,:,i]).reshape(26,26)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        cam[mask_x == 0] = 0
        cam_arr = np.concatenate((cam_arr, cam))
    return cam_arr[1:]

def display_activation(cam_arr, prob, wafer): 
    fig, ax = plt.subplots(9, 1, figsize=(50, 50))
    count = 0
    cam_arr[np.percentile(cam_arr, 0.8) > cam_arr] = 0
    for row in range(0,9):
        ax[row].imshow(np.argmax(wafer, axis=2))
        ax[row].imshow(cam_arr[row],cmap='Reds', alpha=0.7)
        ax[row].set_title('class : ' + faulty_case_dict[count]+', prob : {:.4f}'.format(prob[:, count][0]*100) + '%')
        count += 1


# In[72]:


faulty_case_dict[np.argmax(y_test[target_wafer_num])]


# In[73]:


plt.imshow(np.argmax(x_test[target_wafer_num], axis=2))
print('faulty case : {}'.format(faulty_case_dict[np.argmax(y_test[target_wafer_num])]))


# - 対象のウエハの画像と不良パターンの表示

# In[74]:


cam_result.shape


# In[75]:


cam_arr = make_cam(cam_result, weight_result)


# In[76]:


prob.shape


# In[77]:


display_activation(cam_arr, prob, x_test[target_wafer_num])


# - 活性化マップの表示
# - ドーナツでは円形状にヒートマップの赤い部分が点在している

# In[ ]:





# In[78]:


import requests

# LINEの設定
path = './lineapi.txt'
with open(path) as f:
    s = f.read()
    line_token = s.rstrip('\n')

# LINEに通知する関数
def line_notify(text):
    url = "https://notify-api.line.me/api/notify"
    data = {"message": text}
    headers = {"Authorization": "Bearer " + line_token}
    proxies = {
        'http': 'http://proxy.uec.ac.jp:8080',
        'https': 'https://proxy.uec.ac.jp:8080',
    }
    requests.post(url, data=data, headers=headers, proxies=proxies)
    
line_notify("学習が終了しました")
line_notify("train:" + str(trainscore) + "\nvali:" + str(valiscore) + "\ntest:" + str(testscore))

