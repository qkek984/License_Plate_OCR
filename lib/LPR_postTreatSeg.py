import os
import errno
import shutil
import cv2
from lib import LPR_postTreatFile
class Segmentation:
    def __init__(self,dir):
        self.segNum = []
        self.segChar = []
        self.segRegion = []
        self.saveFileIndex = 0
        self.dir = dir
        self.filePostTreat= LPR_postTreatFile.FilePostTreat()
        try:
            if not (os.path.isdir('./'+dir)):
                os.makedirs(dir)
            else:
                for root, dirs, files in os.walk('./'+dir):
                    for f in files:
                        os.unlink(os.path.join(root, f))
                    for d in dirs:
                        shutil.rmtree(os.path.join(root, d))
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create directory!!!!!")
                raise
    def makeDir(self,dirName):
        try:
            if not (os.path.isdir('./'+dirName)):
                os.makedirs(dirName)
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create directory!!!!!")
                raise

    def saveSeg(self):
        segLists=[self.segNum, self.segChar, self.segRegion]
        for i, segList in enumerate(segLists):
            for item in segList:
                dirName = self.filePostTreat.replaceFileName(item[1][0], indexMode=False)
                self.makeDir(self.dir+'/' + dirName)
                fileName = "[" + str(self.saveFileIndex) + "]" + dirName
                cv2.imwrite(self.dir+'/' + dirName + "/" + fileName+"_"+ str(int(item[1][1])) + ".png", item[0])
                self.saveFileIndex += 1
                print(fileName, " ", str(int(item[1][1])))
        self.removeSeg()

    def removeSeg(self):
        self.segNum = []
        self.segChar = []
        self.segRegion = []





































##Test
if __name__=='__main__':
    lpr_saveResult= FilePostTreat(dir="result")
    st = "인천94너6742"
    st=lpr_saveResult.replaceFileName(st)
    print(st)
