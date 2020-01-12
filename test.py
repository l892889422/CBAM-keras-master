import math, json, os, sys
import numpy as np
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from utils import lr_schedule
import seResnet50
TRAIN_DIR='G:/python/train'
VALID_DIR='G:/python/val'
SIZE = (224, 224)
BATCH_SIZE = 64       #每次送入的数据
if __name__ == "__main__":
    num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)]) #数量
    num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])

    num_train_steps = math.floor(num_train_samples/BATCH_SIZE)    #样本/批数
    num_valid_steps = math.floor(num_valid_samples/BATCH_SIZE)

    gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)            #图片生成器
    val_gen = keras.preprocessing.image.ImageDataGenerator()

    batches = gen.flow_from_directory(TRAIN_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)
    val_batches = val_gen.flow_from_directory(VALID_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)
    #model=keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling='avg', classes=5)
    model=seResnet50(img_input=(224,224,3),classes_num=5)

    #model = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling='avg', classes=5)        #预训练模型
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
    model.summary()
    # 设置损失函数，优化器，模型在训练和测试时的性能指标
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
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