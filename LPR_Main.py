import cv2
import time
import LPR_option
import LPR_find
import LPR_result
from lib import LPR_pretreatment
from lib import LPR_postTreatResult

class LPR_Main:
    def __init__(self):
        ############################# 동영상 선택 ##################################################

        self.cap = cv2.VideoCapture('output.avi')
  

        ########################## 모델 입력 #########################################
        model_array = {'int': 'models/intModelWeight.h5', 'char': 'models/charModelWeight.h5',
                       'stringV': 'models/stringVModelWeight.h5', 'stringH': 'models/stringHModelWeight.h5'}
        self.recog = LPR_find.Recognition(model_array=model_array)
        ###############################################################################
        self.step = 1
        self.time_1 = time.time()
        self.frames = 0
        self.fps = 0
        self.pret = LPR_pretreatment.pret()
        self.test_mode = LPR_option.option().set()
        ########################## 동영상 녹화 설정 ##################################
        if self.test_mode[14][0] == 1:
            self.recod_path = "output.avi" ### 경로 및 이름
            self.fourcc = cv2.VideoWriter_fourcc(*'DIVX') ### 동영상 형식
            self.out = cv2.VideoWriter('output.avi', self.fourcc, 15.0, (1280, 720), 1) ### 녹화 세부 설정
        #####################################################################################
        self.drawing = False
        self.point1 = (0, 100)
        self.point2 = (1275, 710)
        self.result_lpr = LPR_result.Result_LPR(test_mode=self.test_mode)
        self.resultLP = LPR_postTreatResult.ResultLicensePlates(test_mode=self.test_mode)
        cv2.namedWindow("LIVE")
        cv2.setMouseCallback("LIVE", self.mouse_drawing)


    def recog_lpr(self, plates, img, gray):
        found_lprs = []
        for plate in plates:
            if len(plate) < 4:  # too short
                continue
            try:
                lpText, lpImg, lpScore = self.recog.recognize_chars(plate, gray, self.test_mode, img)
                if self.test_mode[10][0] == 1:
                    print(lpScore)
                if lpText:
                    if self.test_mode[13][0] in [1, 2]:
                        trn_img, M = self.pret.transform(plate, gray)
                        cv2.imshow('transform', trn_img)
                        if self.test_mode[13][0] == 2:
                            cv2.waitKey(0)
                            cv2.destroyWindow('transform')
                    x, y, w, h = self.pret.bounding_box_of_rects(plate)
                    found_lprs.append([lpText, lpImg, lpScore, (x, y, w, h)])
            except:
                continue

        return found_lprs

    def mouse_drawing(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.drawing is False:
                self.drawing = True
                self.point1 = (x, y)
            else:
                self.drawing = False

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing is True:
                self.point2 = (x, y)

    def do(self):
        while (self.cap.isOpened()):
            ret, img = self.cap.read()
            if img is None:
                print('no_frame')
                break
            if ret is None:
                print('ret break')
                break
            ####################################################################

            height, width = img.shape[:2]
            if width > 1280 or height > 720:
                img = cv2.resize(img, (1280, 720), cv2.INTER_AREA)
            elif width < 1280 or height < 720:
                img = cv2.resize(img, (1280, 720), cv2.INTER_CUBIC)
            cpimg = img.copy()
            if self.test_mode[14][0] == 1:
                self.out.write(img)
            if self.point1 and self.point2:
                cv2.rectangle(img, self.point1, self.point2, (0, 255, 0))
            if self.test_mode[0][0] == 1:
                cv2.imshow('LIVE', img)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))
            gray = clahe.apply(gray)

            canny = self.pret.preprocess(gray)
            boxes = self.pret.find_contours(canny, self.point1, self.point2)

            plates = self.pret.find_plates(boxes)

            found_lprs = self.recog_lpr(plates, cpimg, gray)

            # print(len(found_lpr))
            if found_lprs:
                for found_lpr in found_lprs:
                    if self.test_mode[2][0] in [2, 3]:
                        print(found_lpr[0])
                    #self.final_result = self.result_lpr.decide(found_lpr)
                    self.resultLP.appendCandidateLP(found_lpr, self.step)
                    if self.test_mode[1][0] in [1, 2]:
                        x, y, w, h = found_lpr[3]
                        cv2.rectangle(img, (x - 1, y - 1), (x + w + 1, y + h + 1), (0, 0, 255), 1)
                        cv2.imshow('Recognized Plate', img)
                    if self.test_mode[1][0] == 2:
                        cv2.waitKey(0)

            self.resultLP.refresh(self.step)
            #self.result_lpr.refresh()

            self.step += 1
            self.frames += 1

            time_2 = time.time()
            if time_2 - self.time_1 >= 1:
                self.fps = self.frames / (time_2 - self.time_1)
                self.time_1 = time_2
                self.frames = 0

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print('FPS: %.1f' % self.fps)
        self.cap.release()
        if self.test_mode[14][0] == 1:
            self.out.release()
        cv2.destroyAllWindows()

        #######################################


LPR_Main().do()
