import cv2
import numpy as np
import imutils
from numpy import matrix


class green:

    def contect_rect(self, plate, lang, count):
        tmp_array = []
        temp = plate.pop(0)
        tmp = (temp[0], temp[1], temp[2], temp[3])
        for p in plate:
            if temp[0] <= p[0] <= temp[0] + temp[2]:
                tmpY = temp[1] + temp[3]
                pY = p[1] + p[3]
                tmp_1 = min(temp[1], p[1])
                tmp_2 = max(temp[2], (p[0] + p[2]) - temp[0])
                tmp_3 = max((tmpY - temp[1]), (pY - temp[1]))
                tmp = (temp[0], tmp_1, tmp_2, tmp_3)
            else:
                tmp_array.append(p)
        tmp_array.append(tmp)
        if count >= lang - 1:
            if len(tmp_array) == len(plate) + 1:
                return tmp_array
        return self.contect_rect(tmp_array, lang, count + 1)

    def max_plate(self, plate):
        return max(matrix(plate).transpose()[2].getA()[0]), max(matrix(plate).transpose()[3].getA()[0])

    def filter_contour(self, contour, mx, my, mod):
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        if mod == 'sub':
            return 0.1 <= aspect_ratio <= 1.2 and my * 0.5 < h < my and mx * 0.9 <= w <= mx * 1.5
        else:
            return 0.1 <= aspect_ratio <= 1.2 and my * 0.3 < h < my and mx * 0.3 <= w <= mx * 1.1

    def find_Contours(self, lpr_img, green_type):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        lpr_img = clahe.apply(lpr_img)
        if green_type == 'sub':
            lpr_img = cv2.GaussianBlur(lpr_img, (5, 5), 0)
            lpr_img = cv2.erode(lpr_img, np.ones((2, 2), np.uint8), iterations=1)
            lpr_img = cv2.dilate(lpr_img, np.ones((3, 3), np.uint8), iterations=2)

        elif green_type in ['a', 'b']:
            lpr_img = cv2.GaussianBlur(lpr_img, (3, 3), 0)
            lpr_img = cv2.erode(lpr_img, np.ones((2, 2), np.uint8), iterations=1)
            lpr_img = cv2.dilate(lpr_img, np.ones((3, 2), np.uint8), iterations=1)

        lpr_img = imutils.auto_canny(lpr_img)
        lpr_img = cv2.copyMakeBorder(lpr_img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        lpr_img = cv2.dilate(lpr_img, np.ones((3, 1), np.uint8), iterations=1)

        if int(cv2.getVersionMajor()) >= 4:
            cntrs, _ = cv2.findContours(lpr_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        else:
            _, cntrs, _ = cv2.findContours(lpr_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return cntrs

    def recog_green(self, img, plate, match, wk):
        top_type = 'sub'
        ps = plate[match]
        pe = plate[match + 3]
        mx, my = self.max_plate(plate)
        sxsy = int(min(ps[1], pe[1]) - max(ps[3], pe[3]) * 0.5)
        sxey = int(min(ps[1], pe[1]) + max(ps[3], pe[3]) * 1.5)
        if sum(wk) > 0:
            mxX = max(matrix(plate).transpose()[2].getA()[0])
            if int(wk[0]) == 1:
                self.sps = int(ps[0] + (ps[2] / 2) - (mxX / 2))
                self.spe = mxX
            if int(wk[1]) == 1:
                self.eps = int(pe[0] + (pe[2] / 2) - (mxX / 2))
                self.epe = mxX
        else:
            self.sps = ps[0]
            self.spe = ps[2]
            self.eps = pe[0]
            self.epe = pe[2]

        self.xx = (self.eps + self.epe - self.sps)
        green_lpr = img[sxsy:sxey, max(0, self.sps - int(self.xx * 0.5)):self.sps]
        if green_lpr.size > 0:
            cntrs = self.find_Contours(green_lpr, top_type)
            green_side = [cv2.boundingRect(cnt) for cnt in cntrs if self.filter_contour(cnt, mx, my, top_type)]
            green_side_img = [(
                              green_lpr[p[1] - 2:p[1] + p[3], p[0] - 2:p[0] + p[2]], green_lpr.shape[1] - (p[0] + p[2]),
                              max(0, self.sps - int(self.xx * 0.5) + p[0]), p[0])
                              for p in green_side]

            return green_side_img, (self.eps + self.epe)
        return None, None

    def green_top(self, img, plate, match, top_type, gap):
        match_set = plate[match:match + 3]
        min_h = min(matrix(match_set).transpose()[1].getA()[0])
        max_h = max(matrix(match_set).transpose()[3].getA()[0])
        top_Y = max(0, int(min_h - int(max_h * 0.5)))
        if top_type == 'a':
            top_img = img[top_Y:min_h, self.sps - gap:self.eps - gap]
            cntrs = self.find_Contours(top_img, top_type)
            mx, my = self.max_plate(plate)
            top = [cv2.boundingRect(cnt) for cnt in cntrs if self.filter_contour(cnt, mx, my, 'top')]
            if len(top) > 3:
                top.sort(key=lambda b: b[0])
                top = self.contect_rect(top, len(top), 0)
                if len(top) > 2:
                    top.sort(key=lambda b: b[0])
                    region_top_minY = min(top[0][1] - 2, top_img[-3][1] - 2)
                    if region_top_minY < 0:
                        region_top_minY = 0
                    region_top_maxY = max(top[0][1] - 2 + top[0][3] - region_top_minY,
                                          top[0][1] - 2 + top[0][3] - region_top_minY)
                    top_region = top_img[region_top_minY:region_top_maxY,
                                 max(0, top[0][0] - 2):top[-3][0] + top[-3][2] + 2]
                    top_num_1 = top_img[top[-2][1] - 1:top[-2][1] + top[-2][3],
                                top[-2][0] - 4:top[-2][0] + top[-2][2] - 1]
                    top_num_2 = top_img[top[-1][1] - 1:top[-1][1] + top[-1][3],
                                top[-1][0] - 4:top[-1][0] + top[-1][2] - 1]
                    resultY = min(top_Y + region_top_minY, top_Y + top[-2][1] - 1, top_Y + top[-1][1] - 1)

                    return (top_region, top_num_1, top_num_2), resultY

        elif top_type == 'b':
            top_img = img[top_Y:min_h, self.sps + self.spe:self.eps]
            if top_img.size > 0:
                cntrs = self.find_Contours(~top_img, top_type)
                mx, my = self.max_plate(plate)

                top = [cv2.boundingRect(cnt) for cnt in cntrs if self.filter_contour(cnt, mx, my, 'top')]
                if len(top) > 2:
                    top.sort(key=lambda b: b[0])
                    top = self.contect_rect(top, len(top), 0)
                    top.sort(key=lambda b: b[0])

                    if len(top) == 3:
                        resultY = top_Y + min(matrix(top).transpose()[1].getA()[0])
                        green_top_img = [top_img[max(0, p[1] - 2):p[1] - 2 + p[3], max(0, p[0] - 3):p[0] - 3 + p[2]] for p in top]
                        return green_top_img, resultY

        return None, None
