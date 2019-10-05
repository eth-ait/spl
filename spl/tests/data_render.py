"""This script shows how to read and convert our data in tfrecord format into numpy.

If you wish, you can store the same data in numpy for your own purpose.
Similarly, you can use our TFRecordMotionDataset class to read, preprocess, normalize the data and then create batches
to train a pytorch model.
"""
import os
import tensorflow as tf
from data.amass_tf import TFRecordMotionDataset
from visualization.fk import SMPLForwardKinematics
from visualization.render import Visualizer
tf.enable_eager_execution()

# Here we visualize a window of 180 frames from full-length test dataset in rotation matrix format.
DATA_PATH = os.path.join(os.environ["AMASS_DATA"], "rotmat", "test_dynamic", "amass-?????-of-?????")
META_DATA_PATH = os.path.join(os.environ["AMASS_DATA"], "rotmat", "training", "stats.npz")
SAVE_DIR = "./tmp_samples/"

# Create dataset object.
tf_data = TFRecordMotionDataset(data_path=DATA_PATH,
                                meta_data_path=META_DATA_PATH,
                                batch_size=1,
                                shuffle=False,
                                extract_windows_of=180,
                                window_type="from_beginning",
                                num_parallel_calls=4,
                                normalize=False)
data_iter_ = tf_data.get_iterator()
batch = data_iter_.get_next()

# Visualize by using SMPL mesh if available. Otherwise, we use matplotlib to visualize skeleton.
try:
    from external.smpl_py3.smpl_webuser.serialization import load_model
    smpl_model = load_model('../../external/smpl_py3/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl')
except:
    smpl_model = None
    print("SMPL model not available.")

fk_engine = SMPLForwardKinematics()
visualizer = Visualizer(fk_engine, SAVE_DIR, SAVE_DIR, rep="rotmat", smpl_model=smpl_model)

pose_seq = batch["inputs"].numpy()
pose_name = batch["id"].numpy()[0].decode("utf-8")
if smpl_model is None:
    # Visualize skeleton.
    visualizer.visualize_skeleton(pose_seq, pose_name)
else:
    visualizer.visualize_dense_smpl(pose_seq, pose_name)
