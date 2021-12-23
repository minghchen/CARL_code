# coding=utf-8
import glob
import math
import os
import numpy as np
import sys
sys.path.append('.')

from absl import app
from absl import flags
from absl import logging

import scipy.io as sio

from dataset_preparation.dataset_utils import write_seqs_to_tfrecords

import cv2

flags.DEFINE_string('dir', '/home/username/datasets/Penn_Action/', 'Path to videos.')
flags.DEFINE_string('name', 'penn_action', 'Name of the dataset being created. This will'
                    'be used as a prefix.')
flags.DEFINE_string('extension', 'jpg', 'Extension of images.')
flags.DEFINE_string('split', 'train', 'train or val.')
flags.DEFINE_string(
    'label_dir', '/home/username/datasets/penn_action_labels/', 'Provide a corresponding labels file'
    'that stores per-frame or per-sequence labels.')
flags.DEFINE_string('output_dir', '/home/username/datasets/penn_action_tfrecords', 'Output directory where'
                    'tfrecords will be stored.')
flags.DEFINE_integer('vids_per_shard', 200, 'Number of videos to store in a'
                     'shard.')
flags.DEFINE_list(
    'frame_labels', '', 'Comma separated list of descriptions '
    'for labels given on a per frame basis. For example: '
    'winding_up,early_cocking,acclerating,follow_through')
flags.DEFINE_integer('action_label', -1, 'Action label of all videos.')
flags.DEFINE_integer('expected_segments', -1, 'Expected number of segments.')
flags.DEFINE_boolean('rotate', False, 'Rotate videos by 90 degrees before'
                     'creating tfrecords')
flags.DEFINE_boolean('resize', True, 'Resize videos to a given size.')
flags.DEFINE_integer('width', 224, 'Width of frames in the TFRecord.')
flags.DEFINE_integer('height', 224, 'Height of frames in the TFRecord.')
flags.DEFINE_integer('fps', 30, 'Frames per second in video.')

FLAGS = flags.FLAGS


def preprocess(im, rotate, resize, width, height):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if resize:
        im = cv2.resize(im, (width, height))
    if rotate:
        im = cv2.transpose(im)
        im = cv2.flip(im, 1)
    return im


def get_frames_in_folder(path, rotate, resize, width, height):
    """Returns all frames from a video in a given folder.

    Args:
        path: string, directory containing frames of a video.
        rotate: Boolean, if True rotates an image by 90 degrees.
        resize: Boolean, if True resizes images to given size.
        width: Integer, Width of image.
        height: Integer, Height of image.
    Returns:
        frames: list, list of frames in a  video.
    Raises:
        ValueError: When provided directory doesn't exist.
    """
    if not os.path.isdir(path):
        raise ValueError('Provided path %s is not a directory' % path)
    else:
        im_list = sorted(glob.glob(os.path.join(path, '*.%s' % FLAGS.extension)))

    frames = [preprocess(cv2.imread(im), rotate, resize, width, height)
                for im in im_list]
    return frames


def get_name(filename, dir, penn_action=False):
    """Add label to name for Penn Action dataset."""
    if penn_action:
        labels_path = os.path.join(dir, 'labels', '%s.mat' % filename)
        annotation = sio.loadmat(labels_path)
        label = annotation['action'][0]
        return '{}_{}'.format(filename, label), label
    else:
        return filename


def get_timestamps(frames, fps, offset=0.0):
    """Returns timestamps for frames in a video."""
    return [offset + x/float(fps) for x in range(len(frames))]


def create_tfrecords(name, output_dir, dir, label_dir,
                     frame_labels, fps, expected_segments):
    """Create TFRecords from videos in a given path.

    Args:
        name: string, name of the dataset being created.
        output_dir: string, path to output directory.
        videos_dir: string, path to input videos directory.
        label_file: string, JSON file that contains annotations.
        frame_labels: list, list of string describing each class. Class label is
        the index in list.
        fps: integer, frames per second with which the images were extracted.
        expected_segments: int, expected number of segments.
    Raises:
        ValueError: If invalid args are passed.
    """
    if not os.path.exists(output_dir):
        print('Creating output directory:', output_dir)
        os.makedirs(output_dir)

    videos_dir = os.path.join(dir, "frames")

    paths = sorted([os.path.join(videos_dir, f) for f in os.listdir(videos_dir)])

    data = {}
    for split in os.listdir(label_dir):
        if os.path.isfile(os.path.join(label_dir, split)): continue
        for action_label_dir in os.listdir(os.path.join(label_dir, split)):
            for video_id in os.listdir(os.path.join(label_dir, split, action_label_dir)):
                if ".npy" in video_id:
                    data[video_id[:4]] = {"split": split, 
                                "labels": np.load(os.path.join(label_dir, split, action_label_dir, video_id)),
                                "action_label": action_label_dir}

    names_to_seqs = {}
    shard_id = 0
    for i, path in enumerate(paths):
        seq = {}

        video_id = os.path.basename(path)
        vid_name, id_label = get_name(video_id, dir, penn_action=True)
        if id_label in ["strum_guitar", "jump_rope"]:
            continue
        frames = get_frames_in_folder(path, FLAGS.rotate, FLAGS.resize,
                                    FLAGS.width, FLAGS.height)

        if video_id in data:
            video_labels = data[video_id]["labels"]
            split = data[video_id]["split"]
        else:
            print(f"Attention! Labels for {vid_name} don't exist!")
            continue
        
        if split != FLAGS.split:
            continue
        
        seq['video'] = frames
        seq['labels'] = video_labels


        if len(seq['video'])!=len(seq['labels']):
            print(vid_name, len(seq['video']), len(seq['labels']))
            if len(seq['video']) > len(seq['labels']):
                seq['video'] = [seq['video'][math.floor(j*len(seq['video'])/len(seq['labels']))] for j in range(len(seq['labels']))]
            else:
                seq['labels'] = [seq['labels'][math.floor(j*len(seq['labels'])/len(seq['video']))] for j in range(len(seq['video']))]

        names_to_seqs[vid_name] = seq
        if (i + 1) % FLAGS.vids_per_shard == 0 or i == len(paths)-1:
            output_filename = os.path.join(
            output_dir,
            '%s_%s-%s.tfrecord' % (name, FLAGS.split,
                                        str(shard_id)))
            write_seqs_to_tfrecords(output_filename, names_to_seqs,
                                    FLAGS.action_label, frame_labels)

            shard_id += 1
            names_to_seqs = {}


def main(_):
    create_tfrecords(FLAGS.name, FLAGS.output_dir, FLAGS.dir,
                    FLAGS.label_dir, FLAGS.frame_labels, FLAGS.fps,
                    FLAGS.expected_segments)


if __name__ == '__main__':
    app.run(main)
