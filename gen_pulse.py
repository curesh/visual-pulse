import cv2 as cv
import numpy as np
import sys
from scipy import signal
from scipy import stats
from matplotlib import pyplot as plt
import json
import sys

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
        crit_flow = dict( criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
                                     30, 0.001))
        crit_pix = dict( criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
                                     30, 0.01))
        cv.cornerSubPix(im1, corner_a, (win_size, win_size), (-1, -1), **crit_pix)
                        
        out = cv.calcOpticalFlowPyrLK(im1, im2, corner_a, None, (win_size, win_size), 5, **crit_flow)
        corner_b, feature_found, feature_error = out
        return im1, im2, corner_b
    
    def _diffAdjacent(self, data):
        diff = [int(data[i]-data[i-1]) for i in range(1,len(data))]
        diff = np.array(diff)
        return diff.astype(int)
        
    def removeOutliers(self, data):
        total_diff = []
        for i, row in enumerate(data):
            diff_sing = self._diffAdjacent(row)
            total_diff.append(diff_sing)
        total_diff = np.array(total_diff)
        mode_diff, mode_count = stats.mode(total_diff)
        print("Mode count: ", mode_count)
        print("Mode value: ", mode_diff)
        print("Sample feature points row: ", data[0])
        mode_diff = np.squeeze(mode_diff)
        deleted = np.copy(data)
        for i, row in enumerate(total_diff):
            if np.amax(row) > mode_diff[0]:
                deleted = np.delete(deleted, data[i], 0)
                print("removing outlier")
        return deleted
        
        
def main():
    numFrames = 500
    #Json file data
    with open('sample/07-01.json') as f:
        data = json.load(f)
    arr_data = [data["/FullPackage"][i]['Value']['waveform'] for i in range(len(data["/FullPackage"]))]
    
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
        if frame.any() and counter < numFrames:
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
        np.transpose(curr)
        data.append(curr)
    origin = np.transpose(data)

    print("Origin shape before removing outliers: ", origin.shape)
    #origin = pfhm.removeOutliers(origin)
    fr = 60/30.
    size_origin = (int(origin.shape[1]*fr), origin.shape[0])
    processed = cv.resize(origin, size_origin, cv.INTER_CUBIC)
    
    #Filter out 0.8 to 3 Hz using butterworth 5th order filter
    nyquist_freq = 15
    print("Processed shape: ", processed.shape)
    b, a = signal.butter(5, [0.75/nyquist_freq, 5/nyquist_freq], 'bandpass')
    filtered = np.array([signal.filtfilt(b, a, row) for row in processed])
    print("Shape filtered out: ", filtered.shape)
    # plt.plot(filtered[0])
    # plt.show()
    
    #remove 25% percent of features with the largest vector norms.
    norm_filtered = np.prod(filtered, 0)
    indices_sort = sorted(range(len(norm_filtered)), key=lambda i: norm_filtered[i])
    limit = int(0.25*len(norm_filtered))
    clean_filtered = np.copy(filtered)
    indices_sort = np.array(indices_sort)
    print("indices_sort: ", indices_sort.shape)
    print("clean filtered shape: ", clean_filtered.shape)
    print("Norm filtered shape: ", norm_filtered.shape)
    for enum, i in enumerate(reversed(indices_sort)):
        if limit < enum:
            print("Breaking!!!", enum, " ")
            break
        delete_check = filtered[:,i]
        print(delete_check.shape)
        clean_filtered = np.delete(clean_filtered, filtered[:,i], 1)
        print("loop clean shape: ", clean_filtered.shape)
    print(clean_filtered.shape)
    mean = np.empty((0))
    mean, eigenvectors = cv.PCACompute(clean_filtered, mean)
    print("eigan shape: ", eigenvectors.shape)
    filtered = np.transpose(filtered)
    
    
    plt.figure(1)
    s = []
    for i in range(5):
        s.append(filtered.dot(eigenvectors[:,i]))
        # plot_num = 611+i
        # plt.subplot(plot_num)
        # plt.plot(s[i])
    # plt.subplot(616)
    # plt.plot(arr_data[0:numFrames*2])
    # plt.show()
    #print("test shape", test.shape)
    # plt.figure(2)
    max_freqs = []
    freq_pulses = []
    for i in range(5):
        f, Pxx_den = signal.periodogram(s[i], 30)
        freq_pulses.append((f, Pxx_den))
        # plot_num = 511+i
        # plt.subplot(plot_num)
        # plt.plot(f, Pxx_den)
        print(i, ": ", f.shape)
        tot_freq = sum(Pxx_den)
        max_freqs.append((np.amax(Pxx_den)/tot_freq, np.argmax(Pxx_den)))
    best_index = (0, max_freqs[0][0])
    #This for loop is to find the element in the 5 element s array that is most periodic
    for i in range(1, len(max_freqs)):
        if max_freqs[i][0] > best_index[1]:
            best_index = (i, max_freqs[i][0])
    f_pulse = freq_pulses[best_index[0]][0][max_freqs[best_index[0]][1]]
    print("F_pulse: ", f_pulse)
    print("BPM: ", 60/f_pulse)
    peaks, _ = signal.find_peaks(s[best_index[0]], distance = int(60/f_pulse))
    plt.figure(1)
    plt.subplot(211)
    plt.plot(s[best_index[0]])
    plt.plot(peaks, s[best_index[0]][peaks], "x")
    plt.subplot(212)
    plt.plot(arr_data[0:numFrames*2])
    
    plt.show()

    #part 3
    # plt.plot(test)
    # plt.show()
    
    
if __name__ == "__main__":
    main()
    