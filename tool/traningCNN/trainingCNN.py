import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

#############################################
class Training:
        def __init__(self, batchSize, steps, valSteps, ep, outputLabel, dir, modelName ):
            #인자값 할당
            self.steps=steps
            self.ep=ep
            self.outputLabel=outputLabel
            self.modelName=modelName
            self.valSteps=valSteps

            # 랜덤시드 고정시키기
            np.random.seed(3)
            # 1. 데이터 생성하기
            train_datagen = ImageDataGenerator(rescale=1. / 255)

            self.train_generator = train_datagen.flow_from_directory(
                dir,
                target_size=(24, 24),
                color_mode='grayscale',
                batch_size=batchSize,  # 한번에 batch size장씩 훈련
                class_mode='categorical')

            test_datagen = ImageDataGenerator(rescale=1. / 255)

            self.test_generator = test_datagen.flow_from_directory(
                "val_"+dir,
                target_size=(24, 24),
                color_mode='grayscale',
                batch_size=batchSize,#128
                class_mode='categorical')

        def makeModel(self):
            # 2. 모델 구성하기
            model = Sequential()

            model.add(Conv2D(32, kernel_size=(3, 3),
                             activation='relu',
                             input_shape=(24, 24, 1)))
            model.add(Conv2D(32, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            model.add(Flatten())
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(self.outputLabel, activation='softmax'))

            # 3. 모델 학습과정 설정하기
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model

        def training(self):
            model = self.makeModel()# 모델구성 함수 호출
            cb_early_stopping = EarlyStopping(monitor='val_loss', patience=20)
            model_path = '{epoch:d}'+self.modelName
            cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss',
                                            verbose=1, save_best_only=True)
            # 4. 모델 학습시키기
            model.fit_generator(  # batch*step= data set
                self.train_generator,
                steps_per_epoch=self.steps,
                epochs=self.ep,  # 총 몇번 훈련
                validation_data=self.test_generator,
                validation_steps=self.valSteps,
                callbacks=[cb_checkpoint])

            # 6. 모델 평가하기
            print("-- Evaluate --")
            scores = model.evaluate_generator(self.test_generator, steps=5)
            #print(self.test_generator.filenames)
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

            # 7. 모델 사용하기
            print("-- Predict --")
            output = model.predict_generator(self.test_generator, steps=5)
            np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
            print(self.test_generator.class_indices)
            print(output)

            # 8. 모델 저장하기
            model.save(self.modelName)

if __name__=='__main__':
    '''--- 트레이닝 값 조정하기 ---'''
    #Training(batchSize=512, steps=52, valSteps= 16, ep=100, outputLabel=11, dir='dataSet/int/', modelName="intModelWeight.h5").training()
    #Training(batchSize=512, steps=167, valSteps=75, ep=45, outputLabel=40, dir='dataSet/char/', modelName="charModel.h5").training()
    Training(batchSize=512, steps=17, valSteps= 8, ep=50, outputLabel=18, dir='dataSet/stringV/', modelName="stringVModel.h5").training()
    #Training(batchSize=256, steps=7, valSteps= 3, ep=197, outputLabel=18, dir='dataSet/stringH/', modelName="stringHModel.h5").training()
