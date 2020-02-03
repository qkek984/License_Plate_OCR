import cv2
import errno
import glob
import random
import shutil

import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

class Generator:
    def __init__(self):
        pass
    def imageGenerator(self,dir,saveDir,crateImgNum,repeat):
        # 랜덤시드 고정시키기
        np.random.seed(5)

        # 데이터셋 부풀리기 설정
        data_aug_gen = ImageDataGenerator(rescale=1. / 255,
                                          rotation_range=10,
                                          width_shift_range=0.015,
                                          height_shift_range=0.015,
                                          shear_range=0.415,
                                          zoom_range=[0.75, 1.25 ],
                                          horizontal_flip=False,  # 좌우 뒤집기
                                          vertical_flip=False,  # 상하 뒤집기
                                          fill_mode='nearest',
                                          channel_shift_range=20
                                          )

        img = load_img(dir)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        # 이 for는 무한으로 반복되기 때문에 우리가 원하는 반복횟수를 지정하여, 지정된 반복횟수가 되면 빠져나오도록 해야합니다.
        for batch in data_aug_gen.flow(x, batch_size=1, save_to_dir=saveDir, save_prefix='gen'+str(repeat),# 폴더모드시 인덱싱을 통한 각 파일이름 변경
                                       save_format='png'):
            i += 1
            if i > crateImgNum-1:
                break

    def makeDir(self,dirName):
        try:
            if not (os.path.isdir('./'+dirName)):
                os.makedirs(dirName)
            else:
                for root, dirs, files in os.walk('./'+dirName):
                    for f in files:
                        os.unlink(os.path.join(root, f))
                    for d in dirs:
                        shutil.rmtree(os.path.join(root, d))
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create directory!!!!!")
                raise

    def run(self,targetDir, resultDir, iteration):#파일기반 부풀리기
        self.makeDir(resultDir)
        files = glob.glob(targetDir + "/*.*")
        generator = Generator()
        for i, file in enumerate(files):
            generator.imageGenerator(file, resultDir, iteration, i)  # 파일명, 결과물 출력경로 , 부풀리기할 객체 갯수, 반복횟수
            # print("file: ", file)

    def runDir(self,targetDir, resultDir, iteration):#폴더기반 부풀리기
        dirs= self.searchDir(targetDir)
        self.makeDir(resultDir)
        for d in dirs:
            if os.path.isfile(os.path.join(targetDir,d)):
                continue
            inResult= resultDir+"/"+d
            self.makeDir(inResult)
            files = glob.glob(targetDir +"/"+d+ "/*.*")
            generator = Generator()
            for i, file in enumerate(files):
                generator.imageGenerator(file, inResult, iteration, i)  # 파일명, 결과물 출력경로 , 부풀리기할 객체 갯수, 반복횟수
                # print("file: ", file)

    def runDirNum(self,targetDir, resultDir, augNum=200):#폴더기반 갯수지정 부풀리기
        dirs = self.searchDir(targetDir)
        self.makeDir(resultDir)
        for d in dirs:
            if os.path.isfile(os.path.join(targetDir,d)):
                continue
            inResult = resultDir + "/" + d
            self.makeDir(inResult)
            files = glob.glob(targetDir + "/" + d + "/*.*")
            generator = Generator()
            iteration = int(augNum/len(files))
            print("[",d,"] iteration: ",iteration)
            for i, file in enumerate(files):
                generator.imageGenerator(file, inResult, iteration, i)  # 파일명, 결과물 출력경로 , 부풀리기할 객체 갯수, 반복횟수
                # print("file: ", file)

    def searchDir(self, dirname):
        dirnames = os.listdir(dirname)
        dirList=[]
        for filename in dirnames:
            #full_filename = os.path.join(dirname, filename)
            dirList.append(filename)
        return dirList

    def runValDirNum(self,targetDir, resultDir, augNum=200):#폴더기반 갯수지정 부풀리기
        dirs = self.searchDir(targetDir)
        self.makeDir(resultDir)
        self.makeDir("val")#
        augNum= int(augNum*1.5)
        for d in dirs:
            if os.path.isfile(os.path.join(targetDir,d)):
                continue
            inResult = resultDir + "/" + d
            valDir = "val/" + d#
            self.makeDir(inResult)
            self.makeDir(valDir)#
            files = glob.glob(targetDir + "/" + d + "/*.*")
            generator = Generator()
            iteration = int(augNum/len(files))
            print("[",d,"] iteration: ",iteration)
            for i, file in enumerate(files):
                generator.imageGenerator(file, inResult, iteration, i)  # 파일명, 결과물 출력경로 , 부풀리기할 객체 갯수, 반복횟수
                # print("file: ", file)
            ###################################
            tmpFiles = glob.glob(inResult+"/*.*")
            for i in range(int(len(tmpFiles) / 3)):
                randomFile = random.choice(tmpFiles)
                #print(os.path.basename(randomFile))
                shutil.move(randomFile, valDir+"/" + os.path.basename(randomFile))
                #print(randomFile)
                tmpFiles.remove(randomFile)
            ###################################


if __name__=='__main__':
    '''run: 파일기반 / runDir: 폴더기반 / runDirNum: 폴더기반, 부풀리기 갯수지정'''
    gen = Generator()
    #gen.run(targetDir="target", resultDir="result", iteration=10)
    #gen.runDir(targetDir="target", resultDir="result", iteration=10)
    #gen.runDirNum(targetDir="target" , resultDir="result", augNum=900)

    #gen.runValDirNum(targetDir="../traningCNN/dataSet/char" , resultDir="../traningCNN/dataSet/augchar", augNum=2000)
    #gen.runValDirNum(targetDir="../traningCNN/dataSet/int" , resultDir="../traningCNN/dataSet/augint", augNum=5000)
    #gen.runValDirNum(targetDir="../traningCNN/dataSet/string" , resultDir="../traningCNN/dataSet/augstring", augNum=500)
    gen.runValDirNum(targetDir="../traningCNN/dataSet/stringH" , resultDir="../traningCNN/dataSet/augstringH", augNum=100)
