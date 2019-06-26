#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Dec 20 17:39 2016

@author: Denis Tome'
"""

import __init__
import logging
import time

from lifting import PoseEstimator
from lifting.utils import draw_limbs
from lifting.utils import plot_pose

import cv2
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from os.path import dirname, realpath

logger = logging.getLogger('Lifting-Pose:RealTime')
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
    logger.debug('cam read+')
    url = "http://10.28.253.46:8080/video"
    cam = cv2.VideoCapture(url)
    ret_val, image = cam.read()
    # create pose estimator
    image_size = image.shape

    pose_estimator = PoseEstimator(image_size, SESSION_PATH, PROB_MODEL_PATH)

    # load model
    pose_estimator.initialise()
    frame = 0

    while True:
        ret_val, image = cam.read()

        logger.debug('image process+')

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # conversion to rgb

        # estimation
        pose_2d, visibility, pose_3d, flag = pose_estimator.estimate(image)

        # close model
        # pose_estimator.close()
        if flag == 1:
            # Show 2D and 3D poses
            # display_results(image, pose_2d, visibility, pose_3d, frame)

            frame += 1

            # logger.debug('show+')
            # cv2.putText(image,
            #             "FPS: %f" % (1.0 / (time.time() - fps_time)),
            #             (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #             (0, 255, 0), 2)
            # cv2.imshow('tf-pose-estimation result', image)

            # logger.debug("FPS: %f" % (1.0 / (time.time() - fps_time)))
            # fps_time = time.time()
            # if cv2.waitKey(1) == 27:
            #     break
            # logger.debug('finished+')
        else:
            continue

        logger.debug("FPS: %f" % (1.0 / (time.time() - fps_time)))
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')
    cv2.destroyAllWindows()


def display_results(in_image, data_2d, joint_visibility, data_3d, frame):
    """Plot 2D and 3D poses for each of the people in the image."""
    plt.figure()
    draw_limbs(in_image, data_2d, joint_visibility)
    # plt.imshow(in_image)
    plt.axis('off')

    # Show 3D poses
    for single_3D in data_3d:
        # or plot_pose(Prob3dPose.centre_all(single_3D))
        plot_pose(single_3D)

    pngName = 'png/test_{0}.jpg'.format(str(frame).zfill(12))
    plt.savefig(pngName)
    # plt.show()

if __name__ == '__main__':
    import sys
    sys.exit(main())
