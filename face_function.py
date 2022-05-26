import mainvideo
import cv2
import numpy as np
import matplotlib.pylab as plt
import face_recognition
import sys, os
sys.path.append(os.pardir)


def face_library() :
    known_face_encodings = []
    known_face_names = []

    dirname = 'C:/choolcheck/knowns_img'
    files = os.listdir(dirname)
    for filename in files:
        name, ext = os.path.splitext(filename)
        if ext == '.jpg':
            known_face_names.append(name)
            pathname = os.path.join(dirname, filename)
            img = face_recognition.load_image_file(pathname)
            face_encoding = face_recognition.face_encodings(img)[0]
            known_face_encodings.append(face_encoding)

    return known_face_encodings, known_face_names

def put_library(known_face_encodings, known_face_names, filename) :
    dirname = 'C:/choolcheck/knowns_img'
    name, ext = os.path.splitext(filename)
    if ext == '.jpg':
        known_face_names.append(name)
        pathname = os.path.join(dirname, filename)
        img = face_recognition.load_image_file(pathname)
        face_encoding = face_recognition.face_encodings(img)[0]
        known_face_encodings.append(face_encoding)
    return known_face_encodings, known_face_names

def prt_test(compare_img, known_face_encodings, known_face_names):

    try:
        compare_face_encoding = face_recognition.face_encodings(compare_img)[0]
        distances = face_recognition.face_distance(known_face_encodings, compare_face_encoding)
        print(distances)
        min_value = min(distances)
        if min_value < 0.45:
            index = np.argmin(distances)
            test_result = known_face_names[index]
            #test_result = 'Pass'
        else:
            test_result='unknown'

        return test_result

    except:
        test_result = 'again'
        return test_result

def prt_result(compare_img, known_face_encodings, known_face_names):
    
    try:
        compare_face_encoding = face_recognition.face_encodings(compare_img)[0]
        distances = face_recognition.face_distance(known_face_encodings, compare_face_encoding)
        #print(distances)
        min_value = min(distances)
        if min_value < 0.45:
            index = np.argmin(distances)
            test_result = known_face_names[index]
            #test_result = 'Pass'
        else:
            test_result='unknown'

        return test_result

    except:
        test_result = 'unknown'
        return test_result

def img_overlay(top_left, bottom_right, main_img) :
    cv2.imwrite('recognition/before_reshape.jpg', main_img)
    before_img = cv2.imread('recognition/before_reshape.jpg')

    under_face = cv2.imread('recognition/under_face.jpg')
    cover_img = cv2.resize(under_face, (int(bottom_right[0]-top_left[0]), int(bottom_right[1]-top_left[1])))
    height, width, channel = cover_img.shape
    before_img[top_left[1]:top_left[1]+height, top_left[0]:top_left[0]+width] = cover_img
    return before_img

def save_img(top_left, bottom_right, main_img, name) :
    cv2.imwrite('recognition/before_reshape.jpg', main_img)
    before_img = cv2.imread('recognition/before_reshape.jpg')

    under_face = cv2.imread('recognition/under_face.jpg')
    cover_img = cv2.resize(under_face, (int(bottom_right[0]-top_left[0]), int(bottom_right[1]-top_left[1])))
    height, width, channel = cover_img.shape
    before_img[top_left[1]:top_left[1]+height, top_left[0]:top_left[0]+width] = cover_img
    cv2.imwrite('knowns_img/'+name+'.jpg', before_img)
