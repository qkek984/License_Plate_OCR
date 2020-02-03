#!/usr/bin/etc python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import re
import LPR_cnn
import LPR_green
from numpy import matrix
from lib import LPR_pretreatment
from lib import LPR_postTreatSeg

np.set_printoptions(threshold=np.inf)


class Recognition:
    def __init__(self, model_array):
        #################################
        self.int_recog = LPR_cnn.PredictCNN(mode='int', model=model_array['int'])
        self.char_recog = LPR_cnn.PredictCNN(mode='char', model=model_array['char'])
        self.region_v_recog = LPR_cnn.PredictCNN(mode='stringV', model=model_array['stringV'])
        self.region_h_recog = LPR_cnn.PredictCNN(mode='stringH', model=model_array['stringH'])
        #####################################
        self.kernel = np.ones((2, 2), np.uint8)
        self.pret = LPR_pretreatment.pret()
        self.segment = LPR_postTreatSeg.Segmentation(dir="result/seg")
        self.green = LPR_green.green()

    def getScore(self, region=None, char=None, num=None, match=None, lpr_type=None):
        if lpr_type == 1:
            pre = [num[match.start()], num[match.end(1) - 1]]
            bef = num[match.start(3):match.end()]
            num = pre+bef

        scoreList = {"region": region, "char": char, "num": num}
        return scoreList

    def maxScore(self, img, tp):
        if tp == "region":
            a = self.region_h_recog.predict(img)
            b = self.region_h_recog.predict(~img)
        elif tp == "num":
            a = self.int_recog.predict(img)
            b = self.int_recog.predict(~img)
        else:
            a = self.char_recog.predict(img)
            b = self.char_recog.predict(~img)
        if a[1] > b[1]:
            return a
        else:
            return b

    def recognize_numbers(self, img, plate, test):
        found_nums = []
        score_nums = []
        num_plate = []
        for p in plate:
            roi = img[p[1] - 3:p[1] + p[3], p[0]:p[0] + p[2]]
            if len(roi) == 0:
                continue
            if len(roi[0]) == 0:
                continue
            area = self.pret.pre_lang(roi)
            if area is None:
                continue
            nbr = self.maxScore(area, 'num')
            if test[3][0] in [1, 2]:
                print(nbr)
                cv2.imshow('num_roi', area)
            if test[3][0] == 2:
                cv2.waitKey(0)
                cv2.destroyWindow('num_roi')
            if int(nbr[1]) >= test[12][0]:
                found_nums.append(str(nbr[0]))
                score_nums.append(int(nbr[1]))
                num_plate.append([p[0], p[1] - 3, p[2], p[3]])
                if test[11][0] == 1:
                    self.segment.segNum.append([area, nbr])
        nums = ''.join(found_nums)
        return nums, score_nums, num_plate

    def recognize_chars(self, plate, img, test_mode, cpimg):
        nums_l, score_nums, nbr_plate = self.recognize_numbers(img, plate, test_mode)
        match = re.search(r'(\d\d)(.*)(\d\d\d\d)', nums_l)
        nums = ''
        regionScore = None
        if match:
            p1 = nbr_plate[match.end(1) - 1]
            p2 = nbr_plate[match.start(3)]
            roi_han = img[min(p1[1], p2[1]):max(p1[1]+p1[3],p2[1] + p2[3]), p1[0] + p1[2]:p2[0]]
            han_h, han_w = roi_han.shape
            if han_h > 3 and han_w > 3:
                area = self.pret.pre_lang(roi_han)
                if area is None or roi_han.shape[1] > roi_han.shape[0]*1.5 or roi_han.shape[0] > roi_han.shape[1]*1.5:
                    return ''
                if test_mode[4][0] in [1, 2]:
                    cv2.imshow('num_roi', area)
                if test_mode[4][0] == 2:
                    cv2.waitKey(0)
                char = self.char_recog.predict(area)
                c = char[0]
                if char[1] >= test_mode[12][1]:
                    st = match.start()
                    ed = match.end() - 1
                    nums += match.group(1) + c + match.group(3)

                    if c in ['아', '바', '사', '자', '배']:
                        roi_region = img[max(0, nbr_plate[st][1] - 5):nbr_plate[st][1] + nbr_plate[st][3],
                                     max(0, int(nbr_plate[st][0] - nbr_plate[st][3] * 8 / 10)):nbr_plate[st][0] - 2]
                        if roi_region.size > 1:
                            lpr_region = self.region_v_recog.predict(roi_region)
                            region_str = lpr_region[0]
                            regionScore = int(lpr_region[1])
                            if regionScore >= test_mode[12][3]:
                                nums = region_str + match.group(1) + c + match.group(3)

                                if test_mode[8][0] in [1, 2]:
                                    cv2.imshow("region", roi_region)
                                if test_mode[8][0] == 2:
                                    cv2.waitKey(0)

                                final_plate = cpimg[min(nbr_plate[st][1], nbr_plate[ed][1]):max(nbr_plate[st][1] + nbr_plate[st][3], nbr_plate[ed][1] + nbr_plate[ed][3]),
                                              max(0, int(nbr_plate[st][0] - nbr_plate[st][3] * 8 / 10)):nbr_plate[ed][0] + nbr_plate[ed][2]]
                    else:
                        final_plate = cpimg[min(nbr_plate[st][1], nbr_plate[ed][1]):max(nbr_plate[st][1] + nbr_plate[st][3],
                                        nbr_plate[ed][1] + nbr_plate[ed][3]), nbr_plate[st][0]:nbr_plate[ed][0] + nbr_plate[ed][2]]

                    if test_mode[11][0] == 1:
                        try:
                            self.segment.segRegion.append([roi_region, lpr_region])
                        except:
                            pass
                        self.segment.segChar.append([area, char])
                        self.segment.saveSeg()
                    score_array = self.getScore(region=regionScore, char=int(char[1]), num=score_nums, match=match, lpr_type=1)
                    return nums, final_plate, score_array

        else:
            match = re.search(r'\d\d\d\d', nums_l)
            if match:
                gray = cv2.cvtColor(cpimg, cv2.COLOR_RGB2GRAY)
                chk_side = []
                wk = [0, 0]
                if int(nums_l[0]) == 1:
                    wk[0] = 1
                if int(nums_l[3]) == 1:
                    wk[1] = 1
                side_img, rsW = self.green.recog_green(gray, nbr_plate, match.start(), wk)
                if side_img is not None:
                    for side in side_img:
                        if side[0].size > 0:
                            roi_side = cv2.threshold(side[0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                            chk, roi_img = self.pret.remove_margin(roi_side, side[0])
                            if chk:
                                side_char = self.char_recog.predict(~roi_img)
                                if side_char[1] > 300:
                                    chk_side.append((side_char, side[1], side[2]))
                                    if test_mode[7][0] in [1, 2]:
                                        cv2.imshow('roi_side', roi_img)
                                        print(side_char)
                                        if test_mode[7][0] == 2:
                                            cv2.waitKey(0)
                                            cv2.destroyWindow('roi_side')
                if len(chk_side) > 0:
                    last_chk_side = max(chk_side, key=lambda b: b[1])
                    top_type = 'a'
                    top, rsY = self.green.green_top(gray, nbr_plate, match.start(), top_type, last_chk_side[1])
                    if top is not None:
                        region_h = self.maxScore(top[0], 'region')
                        num_1 = self.maxScore(top[1], 'num')
                        num_2 = self.maxScore(top[2], 'num')
                        green_rsult = region_h[0] + num_1[0] + num_2[0] + last_chk_side[0][0] + nums_l[match.start():match.end()]
                        green_rsult = ''.join(green_rsult)
                        Mrs_Y = max(matrix(nbr_plate[match.start():match.end()-1]).transpose()[1].getA()[0])
                        Mrs_H = max(matrix(nbr_plate[match.start():match.end()-1]).transpose()[3].getA()[0])
                        type_A_img = cpimg[rsY:Mrs_Y + Mrs_H, last_chk_side[2]:rsW]
                        if test_mode[6][0] in [1, 2]:
                            cv2.imshow('top_region', top[0])
                            cv2.imshow('top_num1', top[1])
                            cv2.imshow('top_num2', top[2])
                            if test_mode[6][0] == 2:
                                cv2.waitKey(0)
                                cv2.destroyWindow('top_region')
                                cv2.destroyWindow('top_num1')
                                cv2.destroyWindow('top_num2')
                        if test_mode[11][0] == 1:
                            self.segment.segNum.insert(0, [top[2], num_2])
                            self.segment.segNum.insert(0, [top[1], num_1])
                            self.segment.segChar.append([~roi_img, side_char])
                            self.segment.segRegion.append([top[0], region_h])
                            self.segment.saveSeg()

                        score_array = self.getScore(region=region_h[1], char=int(side_char[1]), num=[num_1[1], num_2[1]] + score_nums[match.start():match.end()], match=match, lpr_type=2)

                        return green_rsult, type_A_img, score_array

                else:
                    top_type = 'b'
                    top, rsY = self.green.green_top(gray, nbr_plate, match.start(), top_type, 0)
                    if top is not None:
                        num_1 = self.maxScore(top[0], 'num')
                        if num_1[1] > 500:
                            num_2 = self.maxScore(top[1], 'num')
                            if num_2[1] > 500:
                                top_char = self.maxScore(top[2], 'char')
                                if top_char[1] > 300:
                                    if test_mode[5][0] in [1, 2]:
                                        cv2.imshow('top_num1', top[0])
                                        cv2.imshow('top_num2', top[1])
                                        cv2.imshow('top_char', top[2])
                                        if test_mode[5][0] == 2:
                                            cv2.waitKey(0)
                                            cv2.destroyWindow('top_num1')
                                            cv2.destroyWindow('top_num2')
                                            cv2.destroyWindow('top_char')
                                    Mrs_Y = max(matrix(nbr_plate[match.start():match.end() - 1]).transpose()[1].getA()[0])
                                    Mrs_H = max(matrix(nbr_plate[match.start():match.end() - 1]).transpose()[3].getA()[0])
                                    type_B_img = cpimg[rsY:Mrs_Y + Mrs_H, nbr_plate[match.start()][0]:rsW]
                                    green_rsult = num_1[0] + num_2[0] + top_char[0] + nums_l[match.start():match.end()]
                                    green_rsult = ''.join(green_rsult)

                                    if test_mode[11][0] == 1:
                                        self.segment.segNum.insert(0, [top[2], num_2])
                                        self.segment.segNum.insert(0, [top[1], num_1])
                                        self.segment.segChar.append([~roi_img, top_char])
                                        self.segment.saveSeg()
                                    score_array = self.getScore(char=int(top_char[1]),
                                                                num=[num_1[1], num_2[1]] + score_nums + score_nums[match.start():match.end()], match=match, lpr_type=2)
                                    return green_rsult, type_B_img, score_array

        if test_mode[11][0] == 1:
            self.segment.removeSeg()