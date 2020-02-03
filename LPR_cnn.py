import glob
import cv2
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
import numpy as np
import os
from keras.preprocessing import image
from lib import LPR_info

class PredictCNN:
    def __init__(self, mode, model):
        np.random.seed(3)
        #model type
        if mode =="int":
            self.predictionLabel = LPR_info.getIntLabel()
            lastDense = len(self.predictionLabel)
        elif mode =="char":
            self. predictionLabel = LPR_info.getCharLabel()
            lastDense = len(self.predictionLabel)
        elif mode =="stringV" or mode =="stringH" :
            self.predictionLabel = LPR_info.getStrLabel()
            lastDense = len(self.predictionLabel)
        # model setting
        self.size = (24, 24)
        ##
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=(24, 24, 1)))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(lastDense, activation='relu'))

        self.model.load_weights(model)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def predict(self,inputImg):
        img = cv2.resize(inputImg, self.size, interpolation=cv2.INTER_CUBIC)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        score = self.model.predict(img)
        score = score.squeeze()
        maxScore = max(score)
        predictionIndex = (np.where(score == maxScore))[0][0]
        predictionLabel = self.predictionLabel[predictionIndex]
        if predictionLabel=='_' or predictionLabel=='__':
            maxScore=0
        return [predictionLabel, maxScore]


############# Test #############
if __name__=='__main__':
    mode='stringV'
    prediccnn = PredictCNN(mode=mode, model='models/'+mode+'Model.h5')  # 함수 선언 및 할당

    if mode=='int':
        label = LPR_info.getIntLabelDic()
    elif mode=='char':
        label = LPR_info.getCharLabelDic()
    elif mode=='stringV':
        label = LPR_info.getStrLabelDic()
    elif mode=='stringH':
        label = LPR_info.getStrLabelDic()

    #Test Acc
    target_dir = "tool/traningCNN/val_dataSet/" + mode
    #target_dir = "backup/facedata"
    listDir=os.listdir(target_dir)
    totalError=0
    totalFiles=0
    valNum=100
    for folderName in listDir:
        folderName = str(folderName)
        files = glob.glob(target_dir+"/"+folderName+"/"+ "*.*")
        filesLen = len(files)
        error = 0
        errorList = []
        for i in range(filesLen):
            inputImg = cv2.imread(files[i])
        #for i in range(valNum):
            #inputImg = cv2.imread(random.choice(files))
            #inputImg = cv2.resize(inputImg,(24,24), interpolation=cv2.INTER_CUBIC)
            inputImg = cv2.cvtColor(inputImg, cv2.COLOR_RGB2GRAY)
            prediction = prediccnn.predict(inputImg=inputImg)#핵심
            #print(prediction[0], " : ", prediction[1])
            #if prediction != folderName:
            if prediction[0] != label[folderName]:
                error += 1
                errorList.append([prediction[0], prediction[1]])
                # cv2.imwrite("error/["+str(prediction[0])+"]"+str(prediction[1])+".png",inputImg)

        totalError += error
        totalFiles += filesLen
        print("==============================")
        print("<errorList>"+label[folderName])
        #print(errorList)
        print("<result>")
        print("total : "+str(filesLen))
        print("error : ", error)
        print("acc : ", float(filesLen-error) / float(filesLen) * float(100))
        print("==============================")
    totalAcc = 100-((float(totalError) /float(totalFiles)) * float(100))

    '''
        print("==============================")
        print("<errorList>")
        print(errorList)
        print("<result>")
        print("total : "+str(valNum)+"(random)")
        print("error : ", error)
        print("acc : ", float(valNum-error) / float(valNum) * float(100))
        print("==============================")
    totalAcc = float(valNum*len(listDir)-totalError) / float(valNum*len(listDir)) *float(100)
    '''



    print("totalAcc: ", str(totalAcc))
