import cv2
import numpy as np
import imutils
from numpy import matrix

class pret:
    def __init__(self, min_num_area=100, max_num_area=1000):
        self.min_num_area = min_num_area  # for num 1
        self.max_num_area = max_num_area  # for num 8, 0 etc
        self.last_nums = []
        self.min_no_car_frames = 5  #
        self.no_car_frames = 0
        self.kernel = np.ones((2, 2), np.uint8)
        self.point1 = 0
        self.point2 = 0

    def preprocess(self, img):
        blur = cv2.GaussianBlur(img, (7, 7), 0)
        canny = imutils.auto_canny(blur)
        return canny

    def insidechk(self, point, ax):
        if self.point1[ax] < point < self.point2[ax]:
            return True
        else:
            return False

    def filter_contour(self, contour):
        x, y, w, h = cv2.boundingRect(contour)
        if self.insidechk(x, 0) and self.insidechk(x+w, 0) and self.insidechk(y, 1) and self.insidechk(y+h, 1):
            aspect_ratio = w / h
            return 0.1 <= aspect_ratio <= 0.9 and 14 < h < 60 and 4 < w < h * 4

    def find_contours(self, img, p1, p2):
        self.point1 = (max(0, min(p1[0], p2[0])), max(0, min(p1[1], p2[1])))
        self.point2 = (min(max(p1[0], p2[0]), img.shape[1]), min(max(p1[1], p2[1]), img.shape[0]))
        if int(cv2.getVersionMajor()) >= 4:
            cntrs, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        else:
            _, cntrs, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(cnt) for cnt in cntrs if self.filter_contour(cnt)]
        boxes.sort(key=lambda b: (b[2], b[3]), reverse=True)  # sort by width/height, larger  first.
        boxes.sort(key=lambda b: b[0])  # sort by X pos
        return np.array(boxes)

    @staticmethod
    def is_contain(a, b):
        return a[0] + a[2] >= b[0] + b[2] and a[1] + a[3] >= b[1] + b[3]

    @staticmethod
    def bounding_box_of_rects(rects):
        x1 = min(rects, key=lambda b: b[0])[0]
        y1 = min(rects, key=lambda b: b[1])[1]  # y1
        x2 = max(rects, key=lambda b: b[0] + b[2])
        x2 = x2[0] + x2[2]
        y2 = max(rects, key=lambda b: b[1] + b[3])
        y2 = y2[1] + y2[3]
        return x1, y1, (x2 - x1), (y2 - y1)

    def find_plates(self, boxes):
        candidate_plates = []
        box_idx = list(range(len(boxes) - 1))
        while len(box_idx) > 0:
            i = box_idx.pop(0)
            cplate = [boxes[i]]
            for k in box_idx.copy():
                if np.array_equal(cplate[-1], boxes[k]):
                    box_idx.remove(k)
                    continue
                if self.is_contain(cplate[-1], boxes[k]):  # if this box is inside the last box of cplate.
                    continue
                gap_x = abs(boxes[k][0] - cplate[-1][0]) - cplate[-1][2]
                if gap_x > cplate[0][3] * 2:  # compare with the height of the first box
                    break
                gap_y = abs(boxes[k][1] - cplate[-1][1])
                if gap_y > cplate[0][3] // 2:
                    continue
                diff_height = abs(cplate[-1][3] - boxes[k][3])  # height is too different
                if diff_height > cplate[0][3] // 5:
                    continue
                cplate.append(boxes[k])
            if len(cplate) >= 4:
                if len(candidate_plates) > 0:
                    x, y, w, h = self.bounding_box_of_rects(candidate_plates[-1])
                    x2, y2, w2, h2 = self.bounding_box_of_rects(cplate)
                    if self.is_contain([x, y, w, h], [x2, y2, w2, h2]):
                        continue
                candidate_plates.append(cplate)
        candidate_plates.sort(key=len, reverse=True)
        return candidate_plates

    def pre_lang(self, roi_lang):
        copy_img = roi_lang.copy()
        roi_lang = cv2.threshold(roi_lang, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        chk, roi_img = self.remove_margin(roi_lang, copy_img)
        if chk:
            return roi_img
        else:
            return None

    def remove_margin(self, roi, copyimg):
        roi = cv2.dilate(roi, kernel=np.ones((2, 2), np.uint8), iterations=1)
        height, width = roi.shape[:2]
        for x in range(width):
            if np.sum(roi[:, x]) > 0:
                break
        for y in range(height):
            if np.sum(roi[y, :]) > 0:
                break
        if x >= width - 1 or y >= height - 1:
            return True, roi
        for x2 in range(width - 1, x, -1):
            if np.sum(roi[:, x2]) > 0:
                break
        for y2 in range(height - 1, y, -1):
            if np.sum(roi[y2, :]) > 0:
                break

        roi_img = copyimg[max(0, y - 1):min(height, y2 + 1), max(0, x - 1):min(width, x2 + 1)]

        if roi_img.size > 0:
            return True, roi_img
        return False, None

    def transform(self, p, img):
        h = max(matrix(p).transpose()[3].getA()[0])
        pst_1 = np.float32([
            [p[0][0]-2, p[0][1]-2],
            [p[0][0]-2, p[0][1] + p[0][3]+2],
            [p[-1][0] + p[-1][2] + 2, p[-1][1] + 2],
            [p[-1][0] + p[-1][2] + 2, p[-1][1] + p[-1][3]+2]
            ])

        pst_2 = np.float32([
            [0, 0],
            [0, h],
            [p[-1][0] + p[-1][2] - p[0][0], 0],
            [p[-1][0] + p[-1][2] - p[0][0], h]])

        M = cv2.getPerspectiveTransform(pst_1, pst_2)
        dst = cv2.warpPerspective(img, M, ((p[-1][0] + p[-1][2] - p[0][0]), h))

        return dst, M

