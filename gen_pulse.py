import cv2 as cv
import numpy as np
import sys
from scipy import signal
from scipy import stats
from matplotlib import pyplot as plt

class PFHM:
    def __init__(self, path):
        self.fn_haar = path
        self.haar_cascade = cv.CascadeClassifier(path)

    def find_face(self, img, face, face_frame):
        faces = self._haar_faces(img)
        if len(faces) > 0:
            n = self._find_max_face(faces)
            face = np.array(faces[n])
            face_frame = self._find_process_region(img, face)
            img = self._draw_face(img, face, "face")
            return img, face, face_frame
            
        else:
            face_frame = face_frame[face[0]:(face[0]+face[2]),
                                    face[1]:(face[1]+face[3])]
            return img, face, face_frame

    def process_image(self, img, mode):
        PROCESSIMAGE = 1
        NOPROCESSIMAGE = 2
        if mode == PROCESSIMAGE:
            width = int(img.shape[1]/4*3)
            height = int(img.shape[0]/4*3)
            return cv.resize(img, (width, height), 0, 0, cv.INTER_LINEAR)
        elif mode == NOPROCESSIMAGE:
            return img
        else:
            print("Error: Please pass in a correct mode value")
            sys.exit(1)
        
    def _draw_face(self, img, rect, box_text):
        p1 = (rect[0], rect[1])
        p2 = (rect[0]+rect[2], rect[1]+rect[3])
        cv.rectangle(img, p1, p2, (0, 255, 0), thickness=1)
        pos_x = max(rect[0] - 10, 0)
        pos_y = max(rect[1] - 10, 0)
        cv.putText(img, box_text, (pos_x, pos_y), cv.FONT_HERSHEY_PLAIN,
                   1, (0, 255, 0), 2)
        return img
    def _draw_points(self, img, corners, face):
        corners = np.array(corners)
        for point in corners:
            point = np.array(point)
            p1 = face[0]+int(point[0])
            p2 = face[1]+ int(point[1])
            p1 = int(point[0])
            p2 = int(point[1])
            tup_point = (p1, p2)
            # print(type(tup_point))
            # print(type(p1), " ", type(p2))
            # print("Face_frame: ", face_frame[0])
            # print("point: ", point[0])
            cv.circle(img, tup_point, 1, (0, 255, 0))

    def _find_process_region(self, img, face):
        # Main face region
        face[0] += face[2]/4
        face[2] /= 2
        face[3] = face[3]/10*9
        face_frame = img[face[1]:face[1]+face[3], face[0]:face[0]+face[2]]
        #        return cv.cvtColor(face_frame, cv.COLOR_BGR2GRAY)
        # Remove eye region
        up = np.array([0, 0, face[2], face[3]])
        down = np.array([0, 0, face[2], face[3]])
        up[3] = up[3]*10/45
        out = face_frame[up[1]:up[1]+up[3], up[0]:up[0]+up[2]]
        
        down[1] += down[3]*11/20
        down[3] = down[3]*9/20
        down_frame = face_frame[down[1]:down[1]+down[3], down[0]:down[0]+down[2]]
        out = np.concatenate((out, down_frame), axis=0)
        out = cv.cvtColor(out, cv.COLOR_RGB2GRAY)
        return out

    def _haar_faces(self, img):
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        faces = self.haar_cascade.detectMultiScale(gray, 1.1, 2)
        return faces
    
    def _find_max_face(self, faces):
        n = 0
        max_face = -1
        for iter, (x,y,w,h) in enumerate(faces):
            if max_face < w*h:
                max_face = w*h
                n = iter
        return n
    def perform_LKfilter(self, im1, im2, corner_b, win_size, max_corners):
        # im2 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
        # im1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
        corner_a = cv.goodFeaturesToTrack(im1, max_corners, 0.05, 5.0)
        # MAYBE YOU SHOULD | THE TERM CRITERIA INSTEAD OF +
        crit_flow = dict( criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT,
                                     30, 0.001))
        crit_pix = dict( criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT,
                                     30, 0.01))
        cv.cornerSubPix(im1, corner_a, (win_size, win_size), (-1, -1), **crit_pix)
                        
        out = cv.calcOpticalFlowPyrLK(im1, im2, corner_a, None, (win_size, win_size), 5, **crit_flow)
        corner_b, feature_found, feature_error = out
        return im1, im2, corner_b
    
    def _diffAdjacent(self, data):
        diff = [data[i]-data[i-1] for i in range(1,len(data))]
        diff = np.array(diff)
        return diff.astype(int)
    
    def removeOutliers(self, data):
        for row in data:
            diff = self._diffAdjacent(data)
            max_z = np.amax(stats.zscore(diff))
            if max_z > 1.5:
                np.delete(data, row, 0)
                print("Point deleted")   
        return data         
        
        
def main():
    pfhm = PFHM("resources/haarcascade_frontalface_default.xml")
    WINDOW_NAME="Pulse from Head Motion"; 
    cv.namedWindow(WINDOW_NAME, cv.WINDOW_AUTOSIZE)
    window_x = 300
    window_y = 300
    cv.moveWindow(WINDOW_NAME, window_x, window_y)
    counter = 0
    cap  = cv.VideoCapture('sample/07-01.mp4')
    corners = []
    while(cap.isOpened()):
        face_frame = []
        ret, frame = cap.read()
        if frame.any() and counter < 200:
            frame = pfhm.process_image(frame, 2)
            face = np.array([0, 0, 10, 10])
            frame, face, face_frame = pfhm.find_face(frame, face, face_frame)
            # cv.imshow(str(WINDOW_NAME), face_frame)
            if counter == 0:
                saved_frame = face_frame
                saved_size = (saved_frame.shape[:2])[::-1]
            else:
                corner_b = []
                face_frame = cv.resize(face_frame, saved_size, 0, 0, cv.INTER_LINEAR)
                saved_frame, face_frame, corner_b = pfhm.perform_LKfilter(saved_frame, face_frame, corner_b, 15, 200)
                corner_b = np.array(corner_b)
                for point in corner_b:
                    pfhm._draw_points(face_frame, point, face)
                corners.append(np.squeeze(corner_b))
            cv.imshow(str(WINDOW_NAME), face_frame)
            counter += 1
        else:
            break
        if cv.waitKey(5) == 27:
            #cv.destroyWindow(str(WINDOW_NAME))
            cap.release()
    corners = np.array(corners)
    print(corners.shape)
    data = []
    for i in range(len(corners)):
        curr = []
        for j in range (len(corners[i])):
            curr.append(corners[i][j][1])
        print("Processed :", i)
        np.transpose(curr)
        data.append(curr)
    origin = np.transpose(data)
    plt.plot(origin[0])
    plt.show()
    print("Origin shape before removing outliers: ", origin.shape)
    origin = pfhm.removeOutliers(origin)
    print("Origin shape: ", origin.shape)
    print("Reached 139")
    fr = 250/30.
    size_origin = (int(origin.shape[0]*fr), origin.shape[1])
    processed = [cv.resize(row, size_origin, cv.INTER_CUBIC) for row in origin]
    
    
if __name__ == "__main__":
    main()
    