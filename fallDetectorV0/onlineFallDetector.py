'''
    Author: Adrian Sochaniwsky
    Date: March 24, 2022
    Desc: Performs fall detection. MHI is calcualted from input frames, 
            features are extracted from MHI and passed to a SVM for classification
    Inputs: Dataset Video/Live Webcam
            - Optional: Parameters
    Outputs: Displays algorithm stages
            - Sends a text if a fall was detected and there was no response to the program within 3 mins

    Dataset: https://imvia.u-bourgogne.fr/en/database/fall-detection-dataset-2.html
'''

import cv2 as cv
import argparse
import numpy as np
from datetime import datetime

from helper import MHI_Generator, ContourFeatureExtractor, SVM_classifier

SIZE_W = 320
SIZE_H = 240

def main(args):
    capture = cv.VideoCapture(args.input)
    if not capture.isOpened():
        print('Unable to open: ' + args.input)
        exit(0)

    # If step option is enabled, pause at each frame
    if args.step:
        loop_delay_ms = 0
    else:
        # For live video keep delay low for ~30fps output
        # For videos increase the delay to maintain same output
        if args.input == '0':
            loop_delay_ms = 3
        else:
            loop_delay_ms = 50

    # Init processors for feature extraction
    mhi_processor = MHI_Generator()
    contFeaturesProc = ContourFeatureExtractor()
    contFeaturesProc.loadPCAFromFile(args.save_path)

    # Init the pre-trained classifier
    clf = SVM_classifier()
    clf.loadSvmFromFile(args.save_path)

    if args.output_path != '':
        # Setup output video
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
        out = cv.VideoWriter(f'{args.output_path}/{dt_string}.avi',cv.VideoWriter_fourcc('M','J','P','G'), 15,(SIZE_W, SIZE_H))
        #out = cv.VideoWriter('C:/Users/adrso/dev/4TN4\project/test.avi',cv.VideoWriter_fourcc('M','J','P','G'), 15,(SIZE_W, SIZE_H))
        

    while True:
        _, frame = capture.read()
        if frame is None:
            break

        # Frame manipulation
        # ==============================================================
        # Resize image
        frame = cv.resize(frame, (SIZE_W, SIZE_H), interpolation=cv.INTER_AREA)

        # Flip webcam frame for aesthetic purpose
        if args.input == '0':
            frame = cv.flip(frame, flipCode=1)

        # Create Motion History Image
        # ==============================================================
        mhi_frame = mhi_processor.process(frame, save_batch=True)

        if mhi_frame is None:
            continue

        # Feature extraction and reduction
        # ==============================================================
        features = contFeaturesProc.process(mhi_frame)
        reducedFeatures = contFeaturesProc.dimensionReduction(features=features)

        # Predict class fall/not fall
        # ==============================================================
        fall_detected = clf.predict(reducedFeatures)

        cv.rectangle(frame, (SIZE_W-170, 2), (SIZE_W-10,35), (180,180,180), -1)
        if fall_detected:
            cv.putText(frame, 'Fall Detected!', (SIZE_W-150, 25),
                    cv.FONT_HERSHEY_SIMPLEX, .6 , (0,0,255), 1)
        else:
            cv.putText(frame, 'No Fall', (SIZE_W-150, 25),
                    cv.FONT_HERSHEY_SIMPLEX, .6 , (0,255,0), 1)


        # Display
        # ==============================================================
        h, w = tuple(map(lambda i, j: (i-j)//2, frame.shape[0:2], mhi_frame.shape[0:2]))
        mhi_copy = cv.copyMakeBorder(mhi_frame, h, h, w, w, borderType=cv.BORDER_CONSTANT,value=0)
        result = cv.hconcat([frame, np.dstack((mhi_copy, mhi_copy, mhi_copy)).astype(np.uint8)])
        cv.imshow('Pipeline', result)

        if args.output_path != '':
            # Same the result frame to video
            out.write(frame)

        # Maintain loop speed
        # ==============================================================
        keyboard = cv.waitKey(loop_delay_ms)
        if keyboard == 'q' or keyboard == 27:
            break

    if args.output_path != '':
        # Same the result frame to video
        out.release()
    capture.release()


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser(description='Runs fall detection algorithm on video input')
    parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default=0)
    parser.add_argument('--step', type=int, help='Step through video 1 frame at a time, default False', default=1)
    parser.add_argument('--save_path', required=True, default='', help='Path to save pca weights/values')
    parser.add_argument('--output_path', required=False, default='', help='Path to save pca weights/values')
    args = parser.parse_args()

    main(args)