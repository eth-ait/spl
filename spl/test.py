import tensorflow as tf

tf.app.flags.DEFINE_enum("model_type", "zero_velocity", ["zero_velocity"], "Which model: only `zero_velocity` for now.")
tf.app.flags.DEFINE_enum("data_type", "rotmat", ["rotmat", "aa", "quat"],
                         "Which data representation: rotmat (rotation matrix), aa (angle axis), quat (quaternion).")
tf.app.flags.DEFINE_boolean("use_h36m", False, "Use H36M for training and validation.")
tf.app.flags.DEFINE_boolean("no_normalization", False, "If set, do not use zero-mean unit-variance normalization.")
tf.app.flags.DEFINE_boolean("exhaustive_validation", False, "Use entire validation samples (takes much longer).")

tf.app.flags.DEFINE_integer("seed", 1234, "Seed value.")
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size to use during training.")

tf.app.flags.DEFINE_integer("seq_length_in", 50, "Number of frames to feed into the encoder.")
tf.app.flags.DEFINE_integer("seq_length_out", 10, "Number of frames that the decoder has to predict.")


args = tf.app.flags.FLAGS

print("test")
print(args.model_type)
print(args.data_type)


def main(_):
    pass


if __name__ == "__main__":
    tf.app.run()
