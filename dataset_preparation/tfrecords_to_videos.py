import os
import glob
import pickle
from functools import partial
import tensorflow as tf
import torch
import torchvision

PENN_ACTION_LIST = [
    'baseball_pitch',
    'baseball_swing',
    'bench_press',
    'bowl',
    'clean_and_jerk',
    'golf_swing',
    'jumping_jacks',
    'pushup',
    'pullup',
    'situp',
    'squat',
    'tennis_forehand',
    'tennis_serve'
]

def get_tfrecords(dataset, split, path, per_class=False):
    """Get TFRecord files based on dataset and split."""
    if per_class:
        path_to_tfrecords = os.path.join(path, '*%s*'%split)
        print('Loading %s data from: %s' % (split, path_to_tfrecords))
        tfrecord_files = sorted(glob.glob(path_to_tfrecords))
    else:
        path_to_tfrecords = os.path.join(path, '%s_%s*' % (dataset, split))
        print('Loading %s data from: %s' % (split, path_to_tfrecords))
        tfrecord_files = sorted(glob.glob(path_to_tfrecords))

    if len(tfrecord_files)==0:
        raise ValueError('No tfrecords found at path %s' % path_to_tfrecords)

    return tfrecord_files

def _decode(serialized_example, num_parallel_calls=60):
    """Decode serialized SequenceExample."""

    context_features = {
        'name': tf.io.FixedLenFeature([], dtype=tf.string),
        'len': tf.io.FixedLenFeature([], dtype=tf.int64),
        'label': tf.io.FixedLenFeature([], dtype=tf.int64),
    }

    seq_features = {}

    seq_features['video'] = tf.io.FixedLenSequenceFeature([], dtype=tf.string)

    seq_features['frame_labels'] = tf.io.FixedLenSequenceFeature([], dtype=tf.int64)

    # Extract features from serialized data.
    context_data, sequence_data = tf.io.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=seq_features)

    seq_len = context_data['len']
    seq_label = context_data['label']

    video = sequence_data.get('video', [])
    # Decode the encoded JPEG images
    video = tf.map_fn(
        tf.image.decode_jpeg,
        video,
        parallel_iterations=num_parallel_calls,
        dtype=tf.uint8)
    frame_labels = sequence_data.get('frame_labels', [])
    name = tf.cast(context_data['name'], tf.string)
    return video, frame_labels, seq_label, seq_len, name

def main(split="train", path_to_tfrecords='pouring_tfrecords'):
    dataset_name = path_to_tfrecords.strip('_tfrecords')
    path_to_tfrecords = os.path.join("/home/chenminghao/datasets", path_to_tfrecords)
    tfrecord_files = get_tfrecords(dataset_name, split, path_to_tfrecords)
    num_parallel_calls=60
    decode = partial(_decode, num_parallel_calls=num_parallel_calls)
    dataset = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=num_parallel_calls)
    dataset = dataset.map(decode, num_parallel_calls=num_parallel_calls)

    output_dir = path_to_tfrecords.strip('_tfrecords')
    os.makedirs(output_dir, exist_ok=True)
    video_dir = os.path.join(output_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    if 'pouring' in path_to_tfrecords:
        results = []
        for i, (video, frame_label, _, seq_len, name) in enumerate(dataset):
            name = str(name.numpy(), encoding="utf-8")
            video_file = os.path.join(video_dir, name+'.mp4')
            torchvision.io.write_video(video_file, torch.from_numpy(video.numpy()), fps=25)
            data_dict = {"id": i, "video_file": os.path.join("videos", name+'.mp4'), \
                "frame_label": torch.from_numpy(frame_label.numpy()), \
                "seq_len": int(seq_len.numpy()), "name": name}
            results.append(data_dict)
    elif 'penn_action' in path_to_tfrecords:
        action_to_indices = [[] for _ in PENN_ACTION_LIST]
        result_dataset = []
        id = 0
        for i, (video, frame_label, _, seq_len, name) in enumerate(dataset):
            name = str(name.numpy(), encoding="utf-8")
            if name[5:] in PENN_ACTION_LIST:
                action_label = PENN_ACTION_LIST.index(name[5:])
                video_file = os.path.join(video_dir, name+'.mp4')
                torchvision.io.write_video(video_file, torch.from_numpy(video.numpy()), fps=25)
                data_dict = {"id": id, "video_file": os.path.join("videos", name+'.mp4'), \
                    "frame_label": torch.from_numpy(frame_label.numpy()), \
                    "seq_len": int(seq_len.numpy()), "name": name, "action_label": action_label}
                result_dataset.append(data_dict)
                action_to_indices[action_label].append(id)
                id += 1
        results = (result_dataset, action_to_indices)

    with open(os.path.join(output_dir, split+'.pkl'), 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    main(split="train", path_to_tfrecords='penn_action_tfrecords')
    main(split="val", path_to_tfrecords='penn_action_tfrecords')