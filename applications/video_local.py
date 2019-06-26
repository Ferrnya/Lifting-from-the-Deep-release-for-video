#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Fern
"""

import __init__
import logging
import time

from lifting import PoseEstimator
from lifting.utils import draw_limbs
from lifting.utils import plot_pose

import cv2
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from os.path import dirname, realpath
import redis
import pymysql
import json

logger = logging.getLogger('Lifting-Pose:Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

DIR_PATH = dirname(realpath(__file__))
PROJECT_PATH = realpath(DIR_PATH + '/..')
IMAGE_FILE_PATH = PROJECT_PATH + '/data/images/test1.png'
SAVED_SESSIONS_DIR = PROJECT_PATH + '/data/saved_sessions'
SESSION_PATH = SAVED_SESSIONS_DIR + '/init_session/init'
PROB_MODEL_PATH = SAVED_SESSIONS_DIR + '/prob_model/prob_model_params.mat'

def main():
    fps_time = 0
    # logger.debug('cam read+')
    # url = "http://10.28.253.46:8080/video"
    # cam = cv2.VideoCapture(url)
    conn = pymysql.connect(host="xxxx", port=3306, user="xxxx", password="xxxx",
                           db="xxxx", charset='utf8')
    rc = redis.StrictRedis(host="xxxx", port="6379", db=0, decode_responses=True, password="xxxx")

    cap = cv2.VideoCapture('/path/to/local/video')
    ret_val, image = cap.read()
    # create pose estimator
    image_size = image.shape

    pose_estimator = PoseEstimator(image_size, SESSION_PATH, PROB_MODEL_PATH)

    # load model
    pose_estimator.initialise()
    frame = 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    file_output_dir = '/path/to/output_dir'
    result_dir = '/path/to/result_dir'
    file = []

    if cap.isOpened():
        while True:
            ret_val, image = cap.read()
            if ret_val == True:
                logger.debug('image process+')
                fileName = 'test.txt'

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # conversion to rgb

                # estimation
                pose_2d, visibility, pose_3d, flag = pose_estimator.estimate(image)

                # close model
                # pose_estimator.close()
                if flag == 1:
                    # Show 2D and 3D poses
                    # display_results(image, pose_2d, visibility, pose_3d, frame)
                    dc = []
                    if (pose_3d.size == 51):
                        pose_3d_output = pose_3d.reshape([3, 17])
                    else:
                        pose_3d_output = np.zeros([3, 17], float)
                    for i in range(0, 17):
                        d = pose_3d_output[:, i].tolist()
                        human = ' '.join('%f' % id for id in d)
                        dc.append(str(i))
                        dc.append(human)
                    dc = ' '.join(dc)
                    dc.strip()
                    file.append(dc)
                    file.append(';')

                    frame += 1

                else:
                    continue
            else:
                break

            if cv2.waitKey(1) == 27:
                break
            logger.debug('finished+')
    cap.release()
    # f.close()
    file = ''.join(file)

    # 写入数据库
    cursor = conn.cursor()

    usertoken = '123'
    sql = "insert into file_data(user_token,filename,task,pose_data,fps) values ('%s','%s','%d','%s','%d')" % (usertoken, fileName, 3, file, fps)

    try:
        cursor.connection.ping()
        cursor.execute(sql)
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(e)

    cursor.close()

    conn.close()

    resultData = {'filename': fileName, 'videoPath': file_output_dir, 'resultPath': result_dir}
    rc.publish("resultChannel", json.dumps(resultData))

    cv2.destroyAllWindows()


def display_results(in_image, data_2d, joint_visibility, data_3d, frame):
    """Plot 2D and 3D poses for each of the people in the image."""
    plt.figure()
    draw_limbs(in_image, data_2d, joint_visibility)
    # plt.imshow(in_image)
    # plt.axis('off')

    # Show 3D poses
    for single_3D in data_3d:
        # or plot_pose(Prob3dPose.centre_all(single_3D))
        plot_pose(single_3D)

    pngName = 'png/test_{0}.jpg'.format(str(frame).zfill(12))
    plt.savefig(pngName)
    plt.close('all')
    # plt.show()

if __name__ == '__main__':
    import sys
    sys.exit(main())
