import math
import os
import time

import cv2
from lib import LPR_postTreatFile

class ResultLicensePlates:
    def __init__(self, test_mode):
        self.candidateLPinfo = []
        self.lastStep = 0
        self.postTreat = LPR_postTreatFile.FilePostTreat(LPdir="result/lp",exceptDir="result/except")
        self.test_mode = test_mode

    def elementAnalysis(self,cLPinfos):
        elementList = [[], [], [], [], [], [], [], []]  # [객체명, 총합스코어, 카운트]
        maxScoreLP = []
        # 리스트 내 요소, 합스코어, 갯수 추출
        for cLPinfo in cLPinfos:  # [0]lpText,[1]lpImg, [2]lpScore, [3]step
            lpText=cLPinfo[0]
            lpImg=cLPinfo[1]
            lpScore=cLPinfo[2]
            if lpScore["region"] == None:  # 지역명이 없을때
                lpScoreReshape = lpScore["num"][0:2] + [lpScore["char"]] + lpScore["num"][2:]
                for i, Elements in enumerate(elementList[1:]):
                    emptyArray = True
                    for j, nowElement in enumerate(Elements):
                        if nowElement and lpText[i] == nowElement[0]:
                            #elementList[i + 1][j][1] += lpScoreReshape[i]
                            elementList[i + 1][j][1] = max(elementList[i + 1][j][1], lpScoreReshape[i])
                            elementList[i + 1][j][2] += 1
                            emptyArray = False
                            break
                    if emptyArray == True:
                        elementList[i + 1].append([lpText[i], lpScoreReshape[i], 1])
            else:  # 지역명 있을때
                lpScoreReshape = [lpScore["region"]]+lpScore["num"][0:2] + [lpScore["char"]] + lpScore["num"][2:]
                for i, Elements in enumerate(elementList):
                    emptyArray = True
                    for j, nowElement in enumerate(Elements):
                        if i > 0 and nowElement and lpText[i + 1] == nowElement[0]:  # 글자 및 숫자저장시
                            #elementList[i][j][1] += lpScoreReshape[i]
                            elementList[i][j][1] = max(elementList[i][j][1],lpScoreReshape[i])
                            elementList[i][j][2] += 1
                            emptyArray = False
                            break
                        elif nowElement and lpText[i:2] == nowElement[0]:  # 지역명 저장시
                            #elementList[i][j][1] += lpScoreReshape[i]
                            elementList[i][j][1] = max(elementList[i][j][1], lpScoreReshape[i])
                            elementList[i][j][2] += 1
                            emptyArray = False
                            break
                    if emptyArray == True:
                        if i == 0:
                            elementList[i].append([lpText[i:2], lpScoreReshape[i], 1])
                        else:
                            elementList[i].append([lpText[i + 1], lpScoreReshape[i], 1])
            # 통합스코어 비교
            lpScoreTotal = int(sum(lpScoreReshape) / len(lpScoreReshape))
            if not maxScoreLP:
                maxScoreLP.append([lpText, lpScoreTotal, lpScore["region"], lpImg])
            elif maxScoreLP[0][1] < lpScoreTotal:
                maxScoreLP[0] = [lpText, lpScoreTotal, lpScore["region"], lpImg]
        return elementList, maxScoreLP[0]  # elementList: [8][?][객체명, 총합스코어, 카운트] / maxScoreLP: [0] maxLPtext, [1]totalScore, [2]region, [3]img

    def conversionScore(self,elementList):  # elementList: [8][객체명, 총합스코어, 카운트]
        maxElements = [[], [], [], [], [], [], [], []]
        for i, Elements in enumerate(elementList):
            for nowElement in Elements:
                if nowElement[0]=="__" or nowElement[0]=="_":#anomaly클래스일경우 점수0점
                    nowElement[1] = 0
                #ave = nowElement[1] / nowElement[2]
                if nowElement[2] == 1:
                    nowElement[2] = 0  # 가중치 제거
                #nowTotalScore = int(ave + (ave * 0.4 * nowElement[2]))  # 평균 + 가중치
                nowTotalScore = int(nowElement[1]+(nowElement[1]*0.3*nowElement[2]))
                if not maxElements[i]:
                    maxElements[i].append([nowElement[0], nowTotalScore, nowElement[2]])
                elif nowTotalScore > maxElements[i][0][1]:
                    #print(nowTotalScore, maxElements[i][1])
                    maxElements[i][0] = [nowElement[0], nowTotalScore, nowElement[2]]
        return maxElements  # maxElements : [8][최대객체명, 최대스코어, 카운트]

    def regionPostTreat(self, maxElements, maxScoreLP, elementList):
        # maxElements : [8][최대객체명, 최대스코어, 카운트]
        # maxScoreLP: maxScoreLP: [0] maxLPtext, [1]totalScore, [2]region, [3]img
        # elementList: [8][객체명, 총합스코어, 카운트]
        licensePlate = ""
        for maxElement in maxElements:
            if maxElement:
                licensePlate += str(maxElement[0][0])
        if len(licensePlate) == 7:
            if (licensePlate[2] in ['아', '바', '사', '자', '배']) and maxScoreLP[2] != None:
                licensePlate = elementList[0][0] + licensePlate
        else:
            if ((licensePlate[4] in ['아', '바', '사', '자', '배']) is False) and (
                    maxScoreLP[2] == None):  # 영업용 or 지역명있는 구형번호판
                licensePlate = licensePlate[2:]
        return licensePlate

    def scoreAnalysis(self,cLPinfos):
        elementList, maxScoreLP = self.elementAnalysis(cLPinfos)
        # 각 요소별 환산 스코어 계산
        maxElements = self.conversionScore(elementList)
        # 번호판 텍스트 병합
        resultLP = self.regionPostTreat(maxElements, maxScoreLP, elementList)
        return resultLP, maxScoreLP[1], maxScoreLP[3]

    def refresh(self, step):
        if (step - self.lastStep) > 10:
            for i, cLPinfo in enumerate(self.candidateLPinfo):
                resultLP, totalScore,resultImg = self.scoreAnalysis(cLPinfo)

                if totalScore < 2000: #환산스코어가 2000이하면 필터
                    if self.test_mode[9][0] == 1:
                        fileName = self.postTreat.replaceFileName(resultLP)
                        cv2.imwrite("result/except/" + fileName + " _s"+str(totalScore)+'.png', resultImg)
                    continue
                #print("[", str(self.postTreat.saveFileIndex), "]result:", resultLP,"\t",totalScore)
                print("[", str(self.postTreat.saveFileIndex), "]result:", resultLP)
                if self.test_mode[9][0] == 1:
                    fileName = self.postTreat.replaceFileName(resultLP)
                    cv2.imwrite("result/lp/" + fileName + '.png', resultImg)
            self.candidateLPinfo = []  # 리스트 비움

    #######################################################################################
    def hammingDistance(self, text, comparableText):
        hDistance = 0
        hDistanceRightShift = 0
        hDistanceLeftShift = 0
        # 지역명일 경우 지역명 제외
        lenT = len(text)  # 9
        lenC = len(comparableText)  # 7
        if lenT != lenC:
            if lenT > lenC:
                text = text[2:lenT]
                # lenT = len(text)
            else:
                comparableText = comparableText[2:lenC]
                lenC = len(comparableText)
        # 해밍거리 구하기
        for i, item in enumerate(text):
            if item != comparableText[i]:
                hDistance += 1
            if i > 0 and item != comparableText[i - 1]:
                hDistanceRightShift += 1
            if i < (lenC - 1) and item != comparableText[i + 1]:
                hDistanceLeftShift += 1
        return min(hDistance, hDistanceRightShift, hDistanceLeftShift)

    def imgMatch(self, img, img2):
        h, w, _ = img.shape
        h2, w2, _ = img2.shape
        if h >= h2 and w >= w2:
            img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_CUBIC)
        elif h <= h2 and w <= w2:
            img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)
        elif h >= h2 and w <= w2:
            img2 = cv2.resize(img2, (w2, h), interpolation=cv2.INTER_CUBIC)
            img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)
        elif h <= h2 and w >= w2:
            img2 = cv2.resize(img2, (w2, h), interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_CUBIC)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        match = cv2.matchTemplate(img, img2, cv2.TM_CCOEFF_NORMED)
        (_, score, _, _) = cv2.minMaxLoc(match)
        return score

    def positionDistance(self,position, step, comparablePosition, comparableStep):#position: x y w h
        pStep = comparableStep- step
        if pStep> 3:
            return 1000
        centerA = [position[0] + position[2] / 2, position[1] + position[3] / 2]
        centerB = [comparablePosition[0] + comparablePosition[2] / 2, comparablePosition[1] + comparablePosition[3] / 2]
        #print(centerA,centerB)
        dx = centerB[0] - centerA[0]
        dy = centerB[1] - centerA[1]
        pDistance = int(math.sqrt((dx * dx) + (dy * dy)))
        #print(pDistance,"step:",step,comparableStep)
        try:
            psDistance = abs(pDistance/pStep)
        except:
            psDistance = abs(pDistance)
        #print("ps:", str(psDistance))
        return psDistance

    def appendCandidateLP(self, found_lpr, lpStep):
        lpText = found_lpr[0]
        lpImg = found_lpr[1]
        lpScore = found_lpr[2]
        lpPosition = found_lpr[3]

        if not self.candidateLPinfo:  # 리스트가 빔
            self.candidateLPinfo.append([[lpText, lpImg, lpScore, lpPosition, lpStep]])
        else:  # 리스트가 있음.
            runFlag = True
            for i, cLPinfo in enumerate(self.candidateLPinfo):  # [0][0]lpText,[0][1]lpImg, [0][2]lpScore, [0][3]position, [0][4]step
                hDistandce = self.hammingDistance(cLPinfo[0][0],lpText)  # 해밍거리 측정
                if hDistandce <= 3:
                    self.candidateLPinfo[i].append([lpText, lpImg, lpScore, lpPosition, lpStep])
                    runFlag = False
                    break
                else:  # 이미지 매치율 측정
                    imgMatchValue = self.imgMatch(cLPinfo[0][1],lpImg)
                    if imgMatchValue > 0.8:
                        self.candidateLPinfo[i].append([lpText, lpImg, lpScore, lpPosition, lpStep])
                        runFlag = False
                        break
                    else:  # 위치값 측정
                        lastIndex= len(cLPinfo) - 1
                        pDistance = self.positionDistance(cLPinfo[lastIndex][3],cLPinfo[lastIndex][4],lpPosition, lpStep)
                        if pDistance<150:
                            self.candidateLPinfo[i].append([lpText, lpImg, lpScore, lpPosition, lpStep])
                            runFlag = False
                            break
            if runFlag:
                self.candidateLPinfo.append([[lpText, lpImg, lpScore, lpPosition, lpStep]])
        self.lastStep = lpStep