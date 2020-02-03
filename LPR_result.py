import cv2
import time
from lib import LPR_postTreatFile
from difflib import SequenceMatcher


class Result_LPR:
    def __init__(self, test_mode):
        self.data = []
        self.test_mode = test_mode
        self.postTreat = LPR_postTreatFile.FilePostTreat(LPdir="result/lp",exceptDir="result/except")

    def input_result(self, nums, time, img, total):
        self.data.append([nums, time, img, total, 0])

    def decide(self, result):
        chg = 1
        scores = []
        now = time.time()
        if result[2]['region'] == None:
            result[2]['region'] = 0
        result[2] = result[2]['region'] + result[2]['char'] + sum(result[2]['num'])  # 총합 스코어계산
        if len(result[0]) > 7:
            new = str(result[0][2:])
        else:
            new = result[0]
        if len(self.data) == 0:
            self.input_result(result[0], now, result[1], result[2])
        else:
            self.data.sort(key=lambda b: b[1], reverse=True)
            for i, temp_img in enumerate(self.data):
                if len(temp_img[0]) > 7:
                    tmp_lpr = temp_img[0][2:]
                else:
                    tmp_lpr = temp_img[0]
                if SequenceMatcher(None, new, tmp_lpr).ratio() > 0.70:
                    if result[2] > temp_img[3]:
                        # self.data[i] = (result[0], now, temp_img[2], result[2], temp_img[4])
                        self.data[i][0] = result[0]
                        self.data[i][1] = now
                        self.data[i][3] = result[2]
                    else:
                        self.data[i][1] = now
                    chg = 0
                    break
                oh, ow, _ = result[1].shape
                th, tw, _ = temp_img[2].shape
                if oh > th or ow > tw:
                    temp = cv2.resize(temp_img[2], (ow, oh), interpolation=cv2.INTER_CUBIC)
                elif oh < th or ow < tw:
                    temp = cv2.resize(temp_img[2], (ow, oh), interpolation=cv2.INTER_AREA)
                else:
                    temp = temp_img[2]
                match = cv2.matchTemplate(result[1], temp, cv2.TM_CCOEFF_NORMED)
                (_, score, _, _) = cv2.minMaxLoc(match)
                if score >= 0.70:
                    if result[2] > temp_img[3]:
                        # self.data[i] = (result[0], now, temp_img[2], result[2], temp_img[4])
                        self.data[i][0] = result[0]
                        self.data[i][1] = now
                    else:
                        self.data[i][1] = now
                    chg = 0
                    break
                scores.append(score)
            if len(scores) >= 1 and chg == 1:
                self.input_result(result[0], now, result[1], result[2])

        return self.data

    def refresh(self):
        now = time.time()
        for i, tmp in enumerate(self.data):
            if now - tmp[1] > 2:
                del self.data[i]
            else:
                if tmp[4] == 10:
                    if self.test_mode[2][0] in [1, 3]:
                        print('Result :', tmp[0], tmp[3])
                    if self.test_mode[9][0] == 1:
                        bresult = self.postTreat.replaceFileName(tmp[0])
                        cv2.imwrite("result/lp/" + bresult + '.png', tmp[2])
                    self.data[i][4] += 1
                elif tmp[4] < 10:
                    self.data[i][4] += 1
                else:
                    continue
