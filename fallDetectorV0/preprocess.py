'''
Code ref: https://medium.com/diving-in-deep/fall-detection-with-pytorch-b4f19be71e80
'''

import os
import glob
import argparse
import numpy as np
import cv2 as cv

from helper import MHI_Generator, ContourFeatureExtractor

def count_frames(path):
    cap = cv.VideoCapture(path)
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total

def split_indexes(video_files):
    np.random.seed(42)
    idxs = np.arange(len(video_files))
    np.random.shuffle(idxs)
    counts = [count_frames(video_files[idx]) for idx in idxs]
    splits = np.array([0.7, 0.8])
    train, val = tuple(splits * sum(counts))

    idx1 = 0
    idx2 =0
    tt_count = 0
    for idx, count in enumerate(counts):
        tt_count += count
        if tt_count >= train:
            idx1 = idx + 1
            break
    for idx, count in enumerate(counts[idx1:]):
        tt_count += count
        if tt_count >= val:
            idx2 = idx1 + idx + 1
            break
    return idx1, idx2, idxs

def fall_annotations(video_files):
    falls = []
    for file in video_files:
        file = file.replace('Videos', 'Annotation_files')
        file = file.replace('.avi', '.txt')
        with open(file) as f:
            lines = f.readlines()
            falls.append((int(lines[0]), int(lines[1]))) # (start, stop)
    return falls

def prepare_train_val_test(location, src_dir):
    video_files = glob.glob(f'{src_dir}/{location}/Videos/*')
    video_files = sorted(video_files)
    train_end_idx, val_end_idx, idxs = split_indexes(video_files)
    fall_idxs = fall_annotations(video_files)
    train = [(video_files[idx], fall_idxs[idx]) for idx in idxs[:val_end_idx]]
    #val = [(video_files[idx], fall_idxs[idx]) for idx in idxs[train_end_idx:val_end_idx]]
    test = [(video_files[idx], fall_idxs[idx]) for idx in idxs[val_end_idx:]]
    return train, test

def create_MHI(data, dst, dataset='train'):
    dest_path = f'{dst}/{dataset}'

    os.makedirs(f'{dest_path}/fall', exist_ok=True)
    os.makedirs(f'{dest_path}/not_fall', exist_ok=True)

    for file_path, annotation in data:
        prefix = file_path.split(')')
        num = (prefix[0][-2:]).replace('(', '')
        prefix = prefix[0].split('/')[-2] + '_' + num + '_'
        start_fall, stop_fall = annotation

        cap = cv.VideoCapture(file_path)

        mhi_processor = MHI_Generator()

        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret : break

            img = mhi_processor.process(frame, save_batch=True)
            frame_id = mhi_processor.index

            if img is not None:
                if frame_id >= start_fall and frame_id <= stop_fall:
                    cv.imwrite(f'{dest_path}/fall/{prefix}_{frame_id}.png',img)
                else:
                    cv.imwrite(f'{dest_path}/not_fall/{prefix}_{frame_id}.png',img)

        cap.release()

def create_MHIContourFeatures(dst, dataset='train', pcaSavePath=''):
    base_path = f'{dst}/{dataset}'
    locations = [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]

    fallAndNotFallFeatures = []
    for location in locations:
        print(f'Starting in {location}')
        frame_paths = glob.glob(f'{base_path}/{location}/*.png')
        frame_paths = sorted(frame_paths)

        features = []
        contFeaturesProc = ContourFeatureExtractor(debug=False)
        for path in frame_paths:
            frame = contFeaturesProc.getFrame(path)
            val = contFeaturesProc.process(frame)

            if val is not None:
                features.append(val)

        if not isinstance(features, (np.ndarray)):
            features = np.array(features)

        # Perform PCA Reduction
        contFeaturesProc.computePCA(features, numFeaturesToKeep=None)
        contFeaturesProc.savePCAToFile(pcaSavePath)

        features = contFeaturesProc.dimensionReduction(features)

        # Print Stats
        contFeaturesProc.calculateStats(features, printResult=True)

        # write to a file
        np.savetxt(f'{base_path}/{location}_features.csv', features, delimiter=',',fmt='%.3f')

        # Append for viewing/plotting
        fallAndNotFallFeatures.append(features)

    # 2D Plot (if applicable)
    contFeaturesProc.plot2dScatter(fallAndNotFallFeatures[0], fallAndNotFallFeatures[1])

def main(args):
    src_dir = args['source']
    locations = os.listdir(src_dir)

    for location in locations:
        print(f'Starting in {location}')
        train, test = prepare_train_val_test(location, src_dir)
        datasets = {'train': train,
                    'test': test}
        for key, value in datasets.items():
            create_MHI(value, dst=args['dest'], dataset=key)

    for location in ['train', 'test']:
        create_MHIContourFeatures(dst=args['dest'], dataset=location, pcaSavePath=args['save_path'])

    print('Finished Preprocessing!')

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-s', '--source', required=True, default='data',
                    help='Source data folder')
    ap.add_argument('-d', '--dest', required=True, default='dataset',
                    help='Destination dataset folder')
    ap.add_argument('--save_path', required=True, default='',
                    help='Path to save pca weights/values')

    args = vars(ap.parse_args())
    main(args)