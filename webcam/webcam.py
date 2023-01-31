import cv2


class Webcam:
    def __init__(self):
        self.width = 752
        self.height = 480
        self.h = 0
        self.w = 0
        self.capture_number = 0
        self.start_x = 0
        self.start_y = 0
        self.end_x = 0
        self.end_y = 0
        self.correct_device_found = False

    def calibrate_cam(self):
        color = (0, 255, 0)
        border = 1
        step = 5

        win_name = "calibrate capture device"
        cv2.namedWindow(win_name)
        cv2.moveWindow(win_name, 0, 0)

        while not self.correct_device_found:
            cap = cv2.VideoCapture(self.capture_number)
            if not (cap.isOpened()):
                self.capture_number += 1
                continue
            while True:
                ret, frame = cap.read()

                cv2.imshow(win_name, frame)
                k = cv2.waitKey(10) & 0xff
                if k == 27:
                    self.capture_number += 1
                    break
                elif k == 10 or k == 13:
                    self.correct_device_found = True
                    break
            cap.release()

        cv2.destroyWindow("calibrate capture device")
        cap = cv2.VideoCapture(self.capture_number)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not (cap.isOpened()):
            print("Could not open video device")
        ret, frame = cap.read()
        if ret is False:
            print("No frames were grabbed")

        self.h, self.w = frame.shape[0], frame.shape[1]

        self.start_x = int(self.w / 2) - int(self.w / 10 / 2)
        self.start_y = int(self.h / 2) - int(self.w / 10 / 2)
        self.end_x = int(self.w / 2) + int(self.w / 10 / 2)
        self.end_y = int(self.h / 2) + int(self.w / 10 / 2)

        win_name = "calibrate cam"
        cv2.namedWindow(win_name)
        cv2.moveWindow(win_name, 0, 0)

        while True:
            ret, frame = cap.read()
            frame = cv2.rectangle(frame, (self.start_x, self.start_y), (self.end_x, self.end_y), color, border)

            cv2.imshow(win_name, frame)
            k = cv2.waitKey(10) & 0xff
            if k == 49:
                step = 1
            if k == 50:
                step = 5
            if k == 43:
                self.start_x -= step
                self.start_y -= step
                self.end_x += step * 2
                self.end_y += step * 2
            if k == 45:
                self.start_x += step
                self.start_y += step
                self.end_x -= step * 2
                self.end_y -= step * 2
            if k == 119:
                self.start_y -= step * 5
                self.end_y -= step * 5
            if k == 97:
                self.start_x -= step * 5
                self.end_x -= step * 5
            if k == 115:
                self.start_y += step * 5
                self.end_y += step * 5
            if k == 100:
                self.start_x += step * 5
                self.end_x += step * 5
            if k == 10 or k == 13:
                break

        cv2.destroyAllWindows()
        cap.release()

    def get_stream(self):

        cap = cv2.VideoCapture(self.capture_number)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not (cap.isOpened()):
            print("Could not open video device")
        ret, frame = cap.read()
        if ret is False:
            print("No frames were grabbed")

        while True:

            ret, frame = cap.read()
            frame = frame[self.start_y:self.end_y, self.start_x:self.end_x]
            win_name = "stream"
            cv2.imshow(win_name, frame)
            k = cv2.waitKey(10) & 0xff

            if k == 10 or k == 13:
                break

        cv2.destroyWindow("stream")
        cap.release()

    def get_image(self):
        cap = cv2.VideoCapture(self.capture_number)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not (cap.isOpened()):
            print("Could not open video device")
        ret, frame = cap.read()
        if ret is False:
            print("No frames were grabbed")

        ret, frame = cap.read()

        cap.release()
        return frame[self.start_y:self.end_y, self.start_x:self.end_x]
