import math, json, os, sys
import numpy as np
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from utils import lr_schedule
from keras.preprocessing import image
#猫狗数据正常
#番茄99  35哇哇哇哇哇哇哇哇哇哇哇哇哇哇哇哇哇哇哇

#TRAIN_DIR = os.path.join(DATA_DIR, 'train')
#TRAIN_DIR='G:/python/untitled1/demo/data/train/plantvillage'


# TRAIN_DIR='D:/AI1403/ljy/resnet50/untitled1/demo/data/train/train'
# VALID_DIR='D:/AI1403/ljy/resnet50/untitled1/demo/data/validation/validation'
#print(TRAIN_DIR)val
#VALID_DIR = os.path.join(DATA_DIR, 'validation')
#VALID_DIR='G:/python/untitled1/demo/data/validation/test'

TRAIN_DIR='D:/AI1403/ljy/数据集/train'
VALID_DIR='D:/AI1403/ljy/数据集/val'
# TRAIN_DIR='G:/python/train'
# VALID_DIR='G:/python/val'
SIZE = (224, 224)
BATCH_SIZE = 64       #每次送入的数据


if __name__ == "__main__":
    num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)]) #数量
    num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])

    num_train_steps = math.floor(num_train_samples/BATCH_SIZE)    #样本/批数
    num_valid_steps = math.floor(num_valid_samples/BATCH_SIZE)

    gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)            #图片生成器
    val_gen = keras.preprocessing.image.ImageDataGenerator()
    # val_gen = keras.preprocessing.image.ImageDataGenerator( # set input mean to 0 over the dataset
    #     featurewise_center=False,
    #     # set each sample mean to 0
    #     samplewise_center=False,
    #     # divide inputs by std of dataset
    #     featurewise_std_normalization=False,
    #     # divide each input by its std
    #     samplewise_std_normalization=False,
    #     # apply ZCA whitening
    #     zca_whitening=False,
    #     # epsilon for ZCA whitening
    #     zca_epsilon=1e-06,
    #     # randomly rotate images in the range (deg 0 to 180)
    #     rotation_range=90,
    #     # randomly shift images horizontally
    #     width_shift_range=0.1,
    #     # randomly shift images vertically
    #     height_shift_range=0.1,
    #     # set range for random shear
    #     shear_range=0.,
    #     # set range for random zoom
    #     zoom_range=0.,
    #     # set range for random channel shifts
    #     channel_shift_range=0.,
    #     # set mode for filling points outside the input boundaries
    #     fill_mode='nearest',
    #     # value used for fill_mode = "constant"
    #     cval=0.,
    #     # randomly flip images
    #     horizontal_flip=True,
    #     # randomly flip images
    #     vertical_flip=True,
    #     # set rescaling factor (applied before any other transformation)
    #     rescale=1./255,
    #     # set function that will be applied on each input
    #     preprocessing_function=None,
    #     # image data format, either "channels_first" or "channels_last"
    #     data_format=None,
    #     # fraction of images reserved for validation (strictly between 0 and 1)
    #     validation_split=0.0)
    #以文件夹路径为参数,生成经过数据提升/归一化后的数据,在一个无限循环中无限产生batch数据
    #路径、图像尺寸、返回标签数组的形式、是否打乱数据、batch数据的大小，默认为32
    batches = gen.flow_from_directory(TRAIN_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)
    val_batches = val_gen.flow_from_directory(VALID_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)

    #model = keras.applications.resnet50.ResNet50()          #预训练模型
    model = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling='avg', classes=2)        #预训练模型
    classes = list(iter(batches.class_indices))             #用训练好的模型预测时，预测概率序列和Labels的对应关系
    # model.layers.pop()    #弹出模型的最后一层
    #
    # #一个层意味着将其排除在训练之外，即其权重将永远不会更新。
    for layer in model.layers:
        layer.trainable=False
    #
    last = model.layers[-1].output  #输出
    #全连接层 神经元数量和激活函数
    print('神经元数量',len(classes))
    # last = Dropout()(last)
    x = Dense(len(classes), activation="softmax")(last)
    print(model.layers[-1].name)
    model = Model(model.input, x)

    # 设置损失函数，优化器，模型在训练和测试时的性能指标
    model.compile(optimizer=Adam(lr_schedule(0)), loss='categorical_crossentropy', metrics=['accuracy'])
    # for c in batches.class_indices:
    #     classes[batches.class_indices[c]] = c
    # finetuned_model.classes = classes
    #早停法防止过拟合，patience: 当early stop被激活(如发现loss相比上一个epoch训练没有下降)，则经过patience个epoch后停止训练
    #early_stopping = EarlyStopping(patience=10)
    checkpointer = ModelCheckpoint('resnet50_cbam.h5', verbose=1, save_best_only=True)  # 添加模型保存点
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)
    TensorBoardcallback = keras.callbacks.TensorBoard(log_dir='./logs')
    callbacks = [checkpointer, lr_reducer, lr_scheduler,TensorBoardcallback]
    # 拟合模型
    model.fit_generator(batches, steps_per_epoch=num_train_steps, epochs=60,
                                  callbacks=callbacks, validation_data=val_batches,
                                  validation_steps=num_valid_steps)
    #拟合模型
    model.save('resnet50_final.h5')
