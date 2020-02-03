import os
import errno
import shutil
from lib import LPR_info

class FilePostTreat:
    def __init__(self,LPdir=None, exceptDir=None):
        self.charLabel = LPR_info.getCharLabelDic()
        self.strLabel = LPR_info.getStrLabelDic()
        self.saveFileIndex = 0
        self.makeDir(LPdir)
        self.makeDir(exceptDir)

    def makeDir(self,dir):
        if dir != None:
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
    def replaceFileName(self,fileName, indexMode = True):
        for item in self.strLabel.items():
            if item[1] in fileName:
                fileName = fileName.replace(item[1], item[0])
        for item in self.charLabel.items():
            if item[1] in fileName:
                fileName = fileName.replace(item[1], item[0])

        if indexMode == True:
            fileName = "["+str(self.saveFileIndex)+"]"+ fileName
            self.saveFileIndex += 1
        return  fileName

    def roiLicense_plate(self, img, y, h, x, w):
        return img[max(0, y-30):y + h, max(0, x-30):min(img.shape[1], x + w + 30)]
