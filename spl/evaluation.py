import os
import glob
import json
import argparse

import numpy as np
import tensorflow as tf

from common.constants import Constants as C
from data.amass_tf import TFRecordMotionDataset
from model.zero_velocity import ASimpleYetEffectiveBaseline

from visualization.render import Visualizer
from visualization.fk import H36MForwardKinematics
from visualization.fk import SMPLForwardKinematics
from metrics.motion_metrics import MetricsEngine


def load_latest_checkpoint(session, saver, experiment_dir):
    """Restore the latest checkpoint found in `experiment_dir`."""
    ckpt = tf.train.get_checkpoint_state(experiment_dir, latest_filename="checkpoint")

    if ckpt and ckpt.model_checkpoint_path:
        # Check if the specific checkpoint exists
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print("Loading model checkpoint {0}".format(ckpt_name))
        saver.restore(session, ckpt.model_checkpoint_path)
    else:
        raise (ValueError, "Checkpoint {0} does not seem to exist".format(ckpt.model_checkpoint_path))


def get_model_cls(model_type):
    if model_type == C.MODEL_ZERO_VEL:
        return ASimpleYetEffectiveBaseline
    else:
        raise Exception("Unknown model type.")


def create_and_restore_model(session, experiment_dir, data_dir, config, dynamic_test_split):
    model_cls = get_model_cls(config["model_type"])
    window_length = config["source_seq_len"] + config["target_seq_len"]

    if config["use_h36m"]:
        data_dir = os.path.join(data_dir, '../../h3.6m/tfrecords/')

    if dynamic_test_split:
        data_split = "test_dynamic"
    else:
        assert window_length <= 180, "TFRecords are hardcoded with length of 180."
        window_length = 0
        data_split = "test"
    
    test_data_path = os.path.join(data_dir, config["data_type"], data_split, "amass-?????-of-?????")
    meta_data_path = os.path.join(data_dir, config["data_type"], "training", "stats.npz")
    print("Loading test data from " + test_data_path)

    # Create dataset.
    with tf.name_scope("test_data"):
        test_data = TFRecordMotionDataset(data_path=test_data_path,
                                          meta_data_path=meta_data_path,
                                          batch_size=256,
                                          shuffle=False,
                                          extract_windows_of=window_length,
                                          window_type=C.DATA_WINDOW_BEGINNING,
                                          num_parallel_calls=4,
                                          normalize=not config["no_normalization"])
        test_pl = test_data.get_tf_samples()

    # Create model.
    with tf.name_scope(C.TEST):
        test_model = model_cls(
            config=config,
            data_pl=test_pl,
            mode=C.SAMPLE,
            reuse=False)
        test_model.build_graph()
        test_model.summary_routines()

    num_param = 0
    for v in tf.trainable_variables():
        num_param += np.prod(v.shape.as_list())
    print("# of parameters: " + str(num_param))

    # Restore model parameters.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1, save_relative_paths=True)
    load_latest_checkpoint(session, saver, experiment_dir)
    return test_model, test_data


def evaluate(session, test_model, test_data, args, eval_dir, use_h36m):
    test_iter = test_data.get_iterator()

    # Create metrics engine including summaries
    pck_thresholds = C.METRIC_PCK_THRESHS
    if use_h36m:
        fk_engine = H36MForwardKinematics()
        target_lengths = [x for x in C.METRIC_TARGET_LENGTHS_H36M if x <= test_model.target_seq_len]
    else:
        fk_engine = SMPLForwardKinematics()
        target_lengths = [x for x in C.METRIC_TARGET_LENGTHS_AMASS if x <= test_model.target_seq_len]
    
    representation = C.QUATERNION if test_model.use_quat else C.ANGLE_AXIS if test_model.use_aa else C.ROT_MATRIX
    metrics_engine = MetricsEngine(fk_engine,
                                   target_lengths,
                                   force_valid_rot=True,
                                   pck_threshs=pck_thresholds,
                                   rep=representation)
    # create the necessary summary placeholders and ops
    metrics_engine.create_summaries()
    # reset computation of metrics
    metrics_engine.reset()

    def evaluate_model(_eval_model, _eval_iter, _metrics_engine):
        # make a full pass on the validation or test dataset and compute the metrics
        _eval_result = dict()
        _metrics_engine.reset()
        session.run(_eval_iter.initializer)
        try:
            while True:
                # get the predictions and ground-truth values
                prediction, targets, seed_sequence, data_id = _eval_model.sampled_step(session)
                # unnormalize - if normalization is not configured, these calls do nothing
                p = test_data.unnormalize_zero_mean_unit_variance_channel({"poses": prediction}, "poses")
                t = test_data.unnormalize_zero_mean_unit_variance_channel({"poses": targets}, "poses")
                s = test_data.unnormalize_zero_mean_unit_variance_channel({"poses": seed_sequence}, "poses")
                _metrics_engine.compute_and_aggregate(p["poses"], t["poses"])

                # Store each test sample and corresponding predictions with the unique sample IDs.
                for i in range(prediction.shape[0]):
                    _eval_result[data_id[i].decode("utf-8")] = (p["poses"][i], t["poses"][i], s["poses"][i])
        except tf.errors.OutOfRangeError:
            pass
        finally:
            # finalize the computation of the metrics
            final_metrics = _metrics_engine.get_final_metrics()
        return final_metrics, _eval_result

    print("Evaluating test set ...")
    test_metrics, eval_result = evaluate_model(test_model, test_iter, metrics_engine)
    print("Test \t {}".format(metrics_engine.get_summary_string(test_metrics)))
    
    if args.visualize_smpl or args.visualize:
        # visualize some random samples stored in `eval_result` which is a dict id -> (prediction, seed, target)
        video_dir = eval_dir if args.save_video else None
        frames_dir = eval_dir if args.save_frames else None
        visualizer = Visualizer(fk_engine, video_dir, frames_dir, rep=representation)
        n_samples_viz = 30  # TODO change
        selected_idx = [5, 6, 7]  # [5, 6, 7, 19]  # [0, 1, 2, 5, 6, 7, 9, 19, 24, 27] [24, 27]  # for the dynamic split
        rng = np.random.RandomState(42)
        idx = rng.randint(0, len(eval_result), size=n_samples_viz)
    
        sample_keys = [list(sorted(eval_result.keys()))[i] for i in idx]
        for i, k in enumerate(sample_keys):
            if i in selected_idx:
                print("Visualizing sample " + k)
                if args.visualize:
                    visualizer.visualize(eval_result[k][2], eval_result[k][0], eval_result[k][1],
                                         title=k + "_i{}".format(i))
                else:
                    visualizer.visualize_dense_smpl(np.concatenate([eval_result[k][2], eval_result[k][1]], axis=0),
                                                    k + "_i{}".format(i))


if __name__ == '__main__':
    # If you would like to quantitatively evaluate a model, then --dynamic_test_split shouldn't be passed. In this case,
    # the model will be evaluated on 180 frame windows extracted from the entire test split. You can still visualize
    # samples. However, number of predicted frames will be less than or equal to 60.
    # If you intend to evaluate/visualize longer predictions, then you should pass --dynamic_test_split which enables
    # using original full-length test sequences. Hence, --seq_length_out can be much longer than 60.
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', required=True, default=None, type=str,
                        help="Experiment ID (experiment timestamp) or comma-separated list of ids.")
    parser.add_argument('--eval_dir', required=False, default=None, type=str,
                        help="Visualization save directory. If not passed, then save_dir is used.")
    parser.add_argument('--save_dir', required=False, default=None, type=str,
                        help="Path to experiments. If not passed, then AMASS_EXPERIMENTS environment variable is used.")
    parser.add_argument('--data_dir', required=False, default=None, type=str,
                        help="Path to data. If not passed, then AMASS_DATA environment variable is used.")
    
    parser.add_argument('--seq_length_in', required=False, type=int, help="Seed sequence length")
    parser.add_argument('--seq_length_out', required=False, type=int, help="Target sequence length")
    parser.add_argument('--batch_size', required=False, default=64, type=int, help="Batch size")
    
    parser.add_argument('--visualize', required=False, action="store_true",
                        help="Visualize ground-truth and predictions side-by-side by using human skeleton.")
    parser.add_argument('--visualize_smpl', required=False, action="store_true",
                        help="Visualize only predictions by using smpl mesh.")
    parser.add_argument('--save_video', required=False, action="store_true",
                        help="Save the model predictions to mp4 videos in the experiments folder.")
    parser.add_argument('--save_frames', required=False, action="store_true",
                        help="Save the model predictions to individual pngs in a temporary folder")
    parser.add_argument('--dynamic_test_split', required=False, action="store_true",
                        help="Test samples are extracted on-the-fly.")

    _args = parser.parse_args()
    if ',' in _args.model_id:
        model_ids = _args.model_id.split(',')
    else:
        model_ids = [_args.model_id]

    # Set experiment directory.
    _save_dir = _args.save_dir if _args.save_dir else os.environ["AMASS_EXPERIMENTS"]
    # Set data paths.
    _data_dir = _args.data_dir if _args.data_dir else os.environ["AMASS_DATA"]
    
    # Run evaluation for each model id.
    for model_id in model_ids:
        try:
            _experiment_dir = glob.glob(os.path.join(_save_dir, model_id + "-*"), recursive=False)[0]
        except IndexError:
            print("Model " + str(model_id) + " is not found in " + str(_save_dir))
            continue

        try:
            tf.reset_default_graph()
            _config = json.load(open(os.path.abspath(os.path.join(_experiment_dir, 'config.json')), 'r'))
            _config["experiment_dir"] = _experiment_dir

            if _args.seq_length_out is not None and _config["target_seq_len"] != _args.seq_length_out:
                print("!!! Prediction length for training and sampling is different !!!")
                _config["target_seq_len"] = _args.seq_length_out

            if _args.seq_length_in is not None and _config["source_seq_len"] != _args.seq_length_in:
                print("!!! Seed sequence length for training and sampling is different !!!")
                _config["source_seq_len"] = _args.seq_length_in

            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                _eval_dir = _experiment_dir if _args.eval_dir is None else _args.eval_dir
                _test_model, _test_data = create_and_restore_model(sess, _experiment_dir, _data_dir, _config,
                                                                   _args.dynamic_test_split)
                evaluate(sess, _test_model, _test_data, _args, _eval_dir, _config["use_h36m"])
                
        except Exception as e:
            print("something went wrong when evaluating model {}".format(model_id))
            raise Exception(e)
