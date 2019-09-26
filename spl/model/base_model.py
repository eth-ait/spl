from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from common.constants import Constants as C


class BaseModel(object):
    def __init__(self, config, data_pl, mode, reuse, **kwargs):
        self.config = config
        self.data_placeholders = data_pl
        self.mode = mode
        self.reuse = reuse
        
        self.source_seq_len = config["source_seq_len"]
        self.target_seq_len = config["target_seq_len"]
        self.batch_size = config["batch_size"]
        
        # Data placeholders.
        self.data_inputs = data_pl[C.BATCH_INPUT]
        self.data_targets = data_pl[C.BATCH_TARGET]
        self.data_seq_len = data_pl[C.BATCH_SEQ_LEN]
        self.data_ids = data_pl[C.BATCH_ID]

        self.is_eval = self.mode == C.SAMPLE
        self.is_training = self.mode == C.TRAIN
        
        
        self.autoregressive_input = config["autoregressive_input"]
        self.residual_velocities = config["residual_velocities"]
        self.residual_velocities_type = config.get("residual_velocities_type", "plus")
        self.residual_velocities_reg = None  # a regularizer in the residual velocity to be added to the loss
        self.angle_loss_type = config["angle_loss_type"]
        self.joint_prediction_model = config["joint_prediction_model"]
        self.grad_clip_by_norm = config["grad_clip_by_norm"]
        self.loss_on_encoder_outputs = config['loss_on_encoder_outputs']
        self.force_valid_rot = config.get('force_valid_rot', False)
        self.output_layer_config = config.get('output_layer', dict())
        self.rot_matrix_regularization = config.get('rot_matrix_regularization', False)
        self.prediction_activation = None if not self.rot_matrix_regularization else tf.nn.tanh
        self.use_quat = config.get('use_quat', False)
        self.use_aa = config.get('use_aa', False)
        self.h36m_martinez = config.get("use_h36m_martinez", False)
        # Model the outputs with Normal distribution.
        self.mle_normal = self.angle_loss_type == C.LOSS_POSE_NORMAL

        # It is always 1 when we report the loss.
        if not self.is_training:
            self.kld_weight = 1.0

        # Defines how to employ structured latent variables to make predictions.
        # Options are
        # (1) "plain": latent samples correspond to joint predictions. dimensions must meet.
        # (2) "separate_joints": each latent variable is transformed into a joint prediction by using separate networks.
        # (3) "fk_joints": latent samples on the forward kinematic chain are concatenated and used as in (2).
        self.joint_prediction_model = config.get('joint_prediction_model', "plain")
        if config.get('use_sparse_fk_joints', False):
            # legacy, only used so that evaluation of old models can still work
            self.joint_prediction_model = "fk_joints_sparse"

        assert self.joint_prediction_model in ["plain", "separate_joints", "fk_joints",
                                               "fk_joints_sparse", "fk_joints_stop_gradients",
                                               "fk_joints_sparse_shared"]

        # Set by the child model class.
        self.outputs_mu = None  # Mu tensor of predicted frames (Normal distribution).
        self.outputs_sigma = None  # Sigma tensor of predicted frames (Normal distribution).
        self.outputs_mu_joints = list()  # List of individual joint predictions.
        self.outputs_sigma_joints = list()  # List of individual joint predictions.

        self.outputs = None  # List of predicted frames. If the model is probabilistic, a sample is drawn first.
        self.prediction_targets = None  # Targets in pose loss term.
        self.prediction_inputs = None  # Inputs that are used to make predictions.
        self.prediction_representation = None  # Intermediate representation of the model to make predictions.
        self.loss = None  # Loss op to be used in training.
        self.learning_rate = None
        self.learning_rate_scheduler = None
        self.gradient_norms = None
        self.parameter_update = None
        self.summary_update = None

        self.loss_summary = None

        self.prediction_norm = None

        # Hard-coded parameters.
        self.JOINT_SIZE = 4 if self.use_quat else 3 if self.use_aa else 9
        self.NUM_JOINTS = 21 if self.h36m_martinez else 15
        self.HUMAN_SIZE = self.NUM_JOINTS*self.JOINT_SIZE
        self.input_size = self.HUMAN_SIZE

        # [(Parent ID, Joint ID, Joint Name), (...)] where each entry in a list corresponds to the joints at the same
        # level in the joint tree.
        self.structure = [[(-1, 0, "l_hip"), (-1, 1, "r_hip"), (-1, 2, "spine1")],
                          [(0, 3, "l_knee"), (1, 4, "r_knee"), (2, 5, "spine2")],
                          [(5, 6, "spine3")],
                          [(6, 7, "neck"), (6, 8, "l_collar"), (6, 9, "r_collar")],
                          [(7, 10, "head"), (8, 11, "l_shoulder"), (9, 12, "r_shoulder")],
                          [(11, 13, "l_elbow"), (12, 14, "r_elbow")]]

        if self.h36m_martinez:
            self.structure = [[(-1, 0, "Hips")],
                              [(0, 1, "RightUpLeg"), (0, 5, "LeftUpLeg"), (0, 9, "Spine")],
                              [(1, 2, "RightLeg"), (5, 6, "LeftLeg"), (9, 10, "Spine1")],
                              [(2, 3, "RightFoot"), (6, 7, "LeftFoot"), (10, 17, "RightShoulder"),
                               (10, 13, "LeftShoulder"), (10, 11, "Neck")],
                              [(3, 4, "RightToeBase"), (7, 8, "LeftToeBase"), (17, 18, "RightArm"), (13, 14, "LeftArm"),
                               (11, 12, "Head")],
                              [(18, 19, "RightForeArm"), (14, 15, "LeftForeArm")],
                              [(19, 20, "RightHand"), (15, 16, "LeftHand")]]

        # Reorder the structure so that we can access joint information by using its index.
        self.structure_indexed = dict()
        for joint_list in self.structure:
            for joint_entry in joint_list:
                joint_id = joint_entry[1]
                self.structure_indexed[joint_id] = joint_entry

        # Setup learning rate scheduler.
        self.global_step = tf.train.get_global_step(graph=None)
        self.learning_rate_decay_type = config.get('learning_rate_decay_type')
        if self.is_training:
            if config.get('learning_rate_decay_type') == 'exponential':
                self.learning_rate = tf.train.exponential_decay(config.get('learning_rate'),
                                                                global_step=self.global_step,
                                                                decay_steps=config.get('learning_rate_decay_steps'),
                                                                decay_rate=config.get('learning_rate_decay_rate'),
                                                                staircase=True)
            elif config.get('learning_rate_decay_type') == 'piecewise':
                self.learning_rate = tf.Variable(float(config.get('learning_rate')),
                                                 trainable=False,
                                                 dtype=dtype,
                                                 name="learning_rate_op")
                self.learning_rate_scheduler = self.learning_rate.assign(self.learning_rate*config.get('learning_rate_decay_rate'))
            elif config.get('learning_rate_decay_type') == 'fixed':
                self.learning_rate = config.get('learning_rate')
            else:
                raise Exception("Invalid learning rate type")

        # Annealing input dropout rate or using fixed rate.
        self.input_dropout_rate = None
        if config.get("input_layer", None) is not None:
            if isinstance(config["input_layer"].get("dropout_rate", 0), dict):
                self.input_dropout_rate = get_decay_variable(global_step=self.global_step,
                                                             config=config["input_layer"].get("dropout_rate"),
                                                             name="input_dropout_rate")
            elif config["input_layer"].get("dropout_rate", 0) > 0:
                self.input_dropout_rate = config["input_layer"].get("dropout_rate")

        self.normalization_var = kwargs.get('var_channel', None)
        self.normalization_mean = kwargs.get('mean_channel', None)

    def build_graph(self):
        self.build_network()
        self.build_loss()

    def build_network(self):
        pass

    def build_loss(self):
        if self.is_eval or not self.loss_on_encoder_outputs:
            predictions_pose = self.outputs[:, -self.target_seq_len:, :]
            targets_pose = self.prediction_targets[:, -self.target_seq_len:, :]
            seq_len = self.target_seq_len
        else:
            predictions_pose = self.outputs
            targets_pose = self.prediction_targets
            seq_len = tf.shape(self.outputs)[1]

        with tf.name_scope("loss_angles"):
            diff = targets_pose - predictions_pose
            if self.angle_loss_type == "quat_l2":
                assert self.use_quat
                # this is equivalent to log(R*R^T)
                loss_per_frame = quaternion_loss(targets_pose, predictions_pose, self.angle_loss_type)
                loss_per_sample = tf.reduce_sum(loss_per_frame, axis=-1)
                loss_per_batch = tf.reduce_mean(loss_per_sample)
                self.loss = loss_per_batch
            elif self.angle_loss_type == C.LOSS_POSE_ALL_MEAN:
                pose_loss = tf.reduce_mean(tf.square(diff))
                self.loss = pose_loss
            elif self.angle_loss_type == C.LOSS_POSE_JOINT_MEAN:
                per_joint_loss = tf.reshape(tf.square(diff), (-1, seq_len, self.NUM_JOINTS, self.JOINT_SIZE))
                per_joint_loss = tf.sqrt(tf.reduce_sum(per_joint_loss, axis=-1))
                per_joint_loss = tf.reduce_mean(per_joint_loss)
                self.loss = per_joint_loss
            elif self.angle_loss_type == C.LOSS_POSE_JOINT_SUM:
                per_joint_loss = tf.reshape(tf.square(diff), (-1, seq_len, self.NUM_JOINTS, self.JOINT_SIZE))
                per_joint_loss = tf.sqrt(tf.reduce_sum(per_joint_loss, axis=-1))
                per_joint_loss = tf.reduce_sum(per_joint_loss, axis=-1)
                per_joint_loss = tf.reduce_mean(per_joint_loss)
                self.loss = per_joint_loss
            elif self.angle_loss_type == C.LOSS_POSE_NORMAL:
                pose_likelihood = logli_normal_isotropic(targets_pose, self.outputs_mu, self.outputs_sigma)
                pose_likelihood = tf.reduce_sum(pose_likelihood, axis=[1, 2])
                pose_likelihood = tf.reduce_mean(pose_likelihood)
                self.loss = -pose_likelihood
            else:
                raise Exception("Unknown angle loss.")

        if self.residual_velocities_reg is not None:
            self.loss += self.residual_velocities_reg

        if self.rot_matrix_regularization:
            with tf.name_scope("output_rot_mat_regularization"):
                rot_matrix_loss = self.rot_mat_regularization(predictions_pose, summary_name="rot_matrix_reg")
                self.loss += rot_matrix_loss

    def optimization_routines(self):
        if self.config["optimizer"] == C.OPTIMIZER_ADAM:
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.config["optimizer"] == C.OPTIMIZER_SGD:
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            raise Exception("Optimization not found.")

        # Gradients and update operation for training the model.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            params = tf.trainable_variables()
            # Gradient clipping.
            gradients = tf.gradients(self.loss, params)
            if self.config.get('grad_clip_by_norm', 0) > 0:
                gradients, self.gradient_norms = tf.clip_by_global_norm(gradients, self.config.get('grad_clip_by_norm'))
            else:
                self.gradient_norms = tf.global_norm(gradients)
            self.parameter_update = optimizer.apply_gradients(grads_and_vars=zip(gradients, params),
                                                              global_step=self.global_step)

    def step(self, session):
        pass

    def sampled_step(self, session):
        pass

    def parse_outputs(self, prediction_dict):
        self.outputs_mu_joints.append(prediction_dict["mu"])
        if self.mle_normal:
            self.outputs_sigma_joints.append(prediction_dict["sigma"])

    def aggregate_outputs(self):
        self.outputs_mu = tf.concat(self.outputs_mu_joints, axis=-1)
        assert self.outputs_mu.get_shape()[-1] == self.HUMAN_SIZE, "Prediction not matching with the skeleton."
        if self.mle_normal:
            self.outputs_sigma = tf.concat(self.outputs_sigma_joints, axis=-1)

    def get_joint_prediction(self, joint_idx=-1):
        """
        Returns the predicted joint value or whole body.
        """
        if joint_idx < 0:  # whole body.
            assert self.outputs_mu is not None, "Whole body is not predicted yet."
            if self.mle_normal:
                return self.outputs_mu + tf.random.normal(tf.shape(self.outputs_sigma))*self.outputs_sigma
            else:
                return self.outputs_mu
        else:  # individual joint.
            assert joint_idx < len(self.outputs_mu_joints), "The required joint is not predicted yet."
            if self.mle_normal:
                return self.outputs_mu_joints[joint_idx] + tf.random.normal(
                    tf.shape(self.outputs_sigma_joints[joint_idx]))*self.outputs_sigma_joints[joint_idx]
            else:
                return self.outputs_mu_joints[joint_idx]

    def traverse_parents(self, output_list, parent_id):
        """
        Collects joint predictions recursively by following the kinematic chain.
        Args:
            output_list:
            parent_id:
        """
        if parent_id >= 0:
            output_list.append(self.get_joint_prediction(parent_id))
            self.traverse_parents(output_list, self.structure_indexed[parent_id][0])

    def traverse_parents_stop_gradients(self, output_list, parent_id, stop_gradients=False):
        """
        Collects joint predictions recursively by following the kinematic chain but optionally stops gradients.
        Args:
            output_list:
            parent_id:
            stop_gradients:

        """
        if parent_id >= 0:
            parent_prediction = self.get_joint_prediction(parent_id)
            if stop_gradients:
                parent_prediction = tf.stop_gradient(parent_prediction)
            output_list.append(parent_prediction)
            # after the first call we always stop gradients as we want gradients to flow only for the direct parent
            self.traverse_parents_stop_gradients(output_list, self.structure_indexed[parent_id][0], stop_gradients=True)

    def build_output_layer(self):
        """
        Builds layers to make predictions.
        """
        with tf.variable_scope('output_layer', reuse=self.reuse):
            if self.joint_prediction_model == "plain":
                self.parse_outputs(self.build_predictions(self.prediction_representation, self.HUMAN_SIZE, "all"))

            elif self.joint_prediction_model == "separate_joints":
                for joint_key in sorted(self.structure_indexed.keys()):
                    parent_joint_idx, joint_idx, joint_name = self.structure_indexed[joint_key]
                    self.parse_outputs(self.build_predictions(self.prediction_representation, self.JOINT_SIZE, joint_name))

            elif self.joint_prediction_model == "fk_joints":
                # each joint receives direct input from each ancestor in the kinematic chain
                for joint_key in sorted(self.structure_indexed.keys()):
                    parent_joint_idx, joint_idx, joint_name = self.structure_indexed[joint_key]
                    joint_inputs = [self.prediction_representation]
                    self.traverse_parents(joint_inputs, parent_joint_idx)
                    self.parse_outputs(self.build_predictions(tf.concat(joint_inputs, axis=-1), self.JOINT_SIZE, joint_name))

            elif self.joint_prediction_model.startswith("fk_joints_sparse"):
                # each joint only receives the direct parent joint as input
                created_non_root_weights = False
                for joint_key in sorted(self.structure_indexed.keys()):
                    parent_joint_idx, joint_idx, joint_name = self.structure_indexed[joint_key]
                    joint_inputs = [self.prediction_representation]
                    if parent_joint_idx >= 0:
                        joint_inputs.append(self.outputs_mu_joints[parent_joint_idx])

                    if self.joint_prediction_model == "fk_joints_sparse_shared":
                        if parent_joint_idx == -1:
                            # this joint has no parent, so create its own layer
                            name = joint_name
                            share_weights = False
                        else:
                            # always share except for the first joint because we must create at least one layer
                            name = "non_root_shared"
                            share_weights = created_non_root_weights
                            if not created_non_root_weights:
                                created_non_root_weights = True
                    else:
                        name = joint_name
                        share_weights = False
                    self.parse_outputs(self.build_predictions(tf.concat(joint_inputs, axis=-1),
                                                              self.JOINT_SIZE, name, share_weights))

            elif self.joint_prediction_model == "fk_joints_stop_gradients":
                # same as 'fk_joints' but gradients are stopped after the direct parent of each joint
                for joint_key in sorted(self.structure_indexed.keys()):
                    parent_joint_idx, joint_idx, joint_name = self.structure_indexed[joint_key]
                    joint_inputs = [self.prediction_representation]
                    self.traverse_parents_stop_gradients(joint_inputs, parent_joint_idx, stop_gradients=False)
                    self.parse_outputs(
                        self.build_predictions(tf.concat(joint_inputs, axis=-1), self.JOINT_SIZE, joint_name))

            else:
                raise Exception("Joint prediction model '{}' unknown.".format(self.joint_prediction_model))

            self.aggregate_outputs()
            pose_prediction = self.outputs_mu

            # Apply residual connection on the pose only.
            if self.residual_velocities:
                # some debugging
                self.prediction_norm = tf.linalg.norm(pose_prediction)
                # pose_prediction = tf.Print(pose_prediction, [tf.shape(pose_prediction)], "shape", summarize=100)
                # pose_prediction = tf.Print(pose_prediction, [tf.linalg.norm(pose_prediction[0])], "norm[0]", summarize=135)
                # pose_prediction = tf.Print(pose_prediction, [pose_prediction[0]], "pose_prediction[0]", summarize=135)
                # pose_prediction = tf.Print(pose_prediction, [self.prediction_inputs[0, 0:tf.shape(pose_prediction)[1], :self.HUMAN_SIZE]], "inputs[0]", summarize=135)
                if self.residual_velocities_type == "plus":
                    pose_prediction += self.prediction_inputs[:, 0:tf.shape(pose_prediction)[1], :self.HUMAN_SIZE]
                elif self.residual_velocities_type == "matmul":
                    # add regularizer to the predicted rotations
                    self.residual_velocities_reg = self.rot_mat_regularization(pose_prediction,
                                                                               summary_name="velocity_rot_mat_reg")
                    # now perform the multiplication
                    preds = tf.reshape(pose_prediction, [-1, 3, 3])
                    inputs = tf.reshape(self.prediction_inputs[:, 0:tf.shape(pose_prediction)[1], :self.HUMAN_SIZE], [-1, 3, 3])
                    preds = tf.matmul(inputs, preds, transpose_b=True)
                    pose_prediction = tf.reshape(preds, tf.shape(pose_prediction))
                else:
                    raise ValueError("residual velocity type {} unknown".format(self.residual_velocities_type))

            # Enforce valid rotations as the very last step, this currently doesn't do anything with rotation matrices.
            # TODO(eaksan) Not sure how to handle probabilistic predictions. For now we use only the mu predictions.
            if self.force_valid_rot:
                pose_prediction = self.build_valid_rot_layer(pose_prediction)
            self.outputs_mu = pose_prediction
            self.outputs = self.get_joint_prediction(joint_idx=-1)

    def rot_mat_regularization(self, rotmats, summary_name="rot_matrix_reg"):
        """
        Computes || R * R^T - I ||_F and averages this over all joints, frames, and batch entries. Note that we
        do not enforce det(R) == 1.0 for now. The average is added to tensorboard as a summary.
        Args:
            rotmats: A tensor of shape (..., k*3*3)
            summary_name: Name for the summary

        Returns:
            The average deviation of all rotation matrices from being orthogonal.
        """
        rot_matrix = tf.reshape(rotmats, [-1, 3, 3])
        n = tf.shape(rot_matrix)[0]
        rrt = tf.matmul(rot_matrix, rot_matrix, transpose_b=True)
        eye = tf.eye(3, batch_shape=[n])
        rot_reg = tf.norm(rrt - eye, ord='fro', axis=(-1, -2))
        rot_reg = tf.reduce_mean(rot_reg)
        tf.summary.scalar(self.mode + "/" + summary_name, rot_reg, collections=[self.mode + "/model_summary"])
        return rot_reg

    def get_closest_rotmats(self, rotmats):
        """
        Finds the rotation matrix that is closest to the inputs in terms of the Frobenius norm. For each input matrix
        it computes the SVD as R = USV' and sets R_closest = UV'.

        WARNING: tf.svd is very slow - use at your own peril.

        Args:
            rotmats: A tensor of shape (N, seq_length, n_joints*9) containing the candidate rotation matrices.

        Returns:
            A tensor of the same shape as `rotmats` containing the closest rotation matrices.
        """
        assert not self.use_quat and not self.use_aa
        # reshape to (N, seq_len, n_joints, 3, 3)
        seq_length = tf.shape(rotmats)[1]
        dof = rotmats.get_shape()[-1].value
        rots = tf.reshape(rotmats, [-1, seq_length, dof//9, 3, 3])

        # add tanh activation function to map to [-1, 1]
        rots = tf.tanh(rots)

        # compute SVD
        # This is problematic when done on the GPU, see https://github.com/tensorflow/tensorflow/issues/13603
        s, u, v = tf.svd(rots, full_matrices=True)
        closest_rot = tf.matmul(u, v, transpose_b=True)
        closest_rot = tf.Print(closest_rot, [closest_rot], "done with SVD")

        # TODO(kamanuel) should we make sure that det == 1?

        raise ValueError("SVD on GPU is super slow, not recommended to use.")
        # return tf.reshape(closest_rot, [-1, seq_length, dof])

    def normalize_quaternions(self, quats):
        """
        Normalizes the input quaternions to have unit length.
        Args:
            quats: A tensor of shape (..., k*4) or (..., 4).

        Returns:
            A tensor of the same shape as the input but with unit length quaternions.
        """
        assert self.use_quat
        last_dim = quats.get_shape()[-1].value
        ori_shape = tf.shape(quats)
        if last_dim != 4:
            assert last_dim % 4 == 0
            new_shape = tf.concat([ori_shape[:-1], [last_dim // 4, 4]], axis=0)
            quats = tf.reshape(quats, new_shape)
        else:
            quats = quats

        mag = tf.sqrt(tf.reduce_sum(quats*quats, axis=-1, keepdims=True))
        quats_normalized = quats / mag

        # if the magnitude is too small, replace the quaternion with the identity
        quat_ident = tf.concat([tf.ones_like(mag)] + [tf.zeros_like(mag)]*3, axis=-1)
        is_zero = tf.less_equal(mag, 1e-16)
        perc_zero =  tf.reduce_sum(tf.cast(is_zero, tf.float32)) / tf.cast(tf.reduce_prod(tf.shape(is_zero)), tf.float32)
        tf.summary.scalar(self.mode + "/n_zero_quats", perc_zero, collections=[self.mode + "/model_summary"])
        is_zero = tf.concat([is_zero]*4, axis=-1)
        quats_normalized = tf.where(is_zero, quat_ident, quats_normalized)

        quats_normalized = tf_tr_utils_assert.assert_normalized(quats_normalized, eps=1e-12)
        quats_normalized = tf.reshape(quats_normalized, ori_shape)
        return quats_normalized

    def build_valid_rot_layer(self, input_):
        """
        Ensures that the given rotations are valid. Can handle quaternion and rotation matrix input.
        Args:
            input_: A tensor of shape (N, seq_length, n_joints*dof) containing the candidate orientations. For
              quaternions `dof` is expected to be 4, otherwise it's expected to be 3*3.

        Returns:
            A tensor of the same shape as `input_` containing valid rotations.
        """
        if self.use_quat:
            # activation function to map to [-1, 1]
            input_t = tf.tanh(input_)

            # monitor the average norm of the quaternions in tensorboard
            qn = tf.reduce_mean(quaternion_norm(input_t))
            tf.summary.scalar(self.mode + "/quat_norm_before", qn, collections=[self.mode + "/model_summary"])

            # now normalize
            return self.normalize_quaternions(input_t)
        elif self.use_aa:
            return input_
        else:
            return self.get_closest_rotmats(input_)

    def build_predictions(self, inputs, output_size, name, share=False):
        """
        Builds dense output layers given the inputs. First, creates a number of hidden layers if set in the config and
        then makes the prediction without applying an activation function.
        Args:
            inputs (tf.Tensor):
            output_size (int):
            name (str):
            share (bool): If true all joints share the same weights.
        Returns:
            (tf.Tensor) prediction.
        """
        hidden_size = self.output_layer_config.get('size', 0)
        num_hidden_layers = self.output_layer_config.get('num_layers', 0)

        prediction = dict()
        current_layer = inputs
        for layer_idx in range(num_hidden_layers):
            with tf.variable_scope('out_dense_' + name + "_" + str(layer_idx), reuse=share or self.reuse):
                current_layer = tf.layers.dense(inputs=current_layer, units=hidden_size, activation=self.activation_fn)

        with tf.variable_scope('out_dense_' + name + "_" + str(num_hidden_layers), reuse=share or self.reuse):
            prediction["mu"] = tf.layers.dense(inputs=current_layer, units=output_size,
                                               activation=self.prediction_activation)

        if self.mle_normal:
            with tf.variable_scope('out_dense_sigma_' + name + "_" + str(num_hidden_layers), reuse=share or self.reuse):
                sigma = tf.layers.dense(inputs=current_layer,
                                        units=output_size,
                                        activation=tf.nn.softplus)
                # prediction["sigma"] = tf.clip_by_value(sigma, 1e-4, 5.0)
                prediction["sigma"] = sigma
        return prediction

    def summary_routines(self):
        # Note that summary_routines are called outside of the self.mode name_scope. Hence, self.mode should be
        # prepended to summary name if needed.
        if self.mle_normal:
            tf.summary.scalar(self.mode + "/likelihood", -self.loss, collections=[self.mode + "/model_summary"])
            tf.summary.scalar(self.mode + "/avg_sigma", tf.reduce_mean(self.outputs_sigma), collections=[self.mode + "/model_summary"])
        elif self.use_quat:
            tf.summary.scalar(self.mode + "/loss_quat", self.loss, collections=[self.mode + "/model_summary"])
        else:
            tf.summary.scalar(self.mode+"/loss", self.loss, collections=[self.mode+"/model_summary"])

        if self.is_training:
            tf.summary.scalar(self.mode + "/learning_rate",
                              self.learning_rate,
                              collections=[self.mode + "/model_summary"])
            tf.summary.scalar(self.mode + "/gradient_norms",
                              self.gradient_norms,
                              collections=[self.mode + "/model_summary"])

        if self.input_dropout_rate is not None and self.is_training:
            tf.summary.scalar(self.mode + "/input_dropout_rate",
                              self.input_dropout_rate,
                              collections=[self.mode + "/model_summary"])

        if self.prediction_norm is not None:
            tf.summary.scalar(self.mode + "/prediction_norm_before_residual",
                              self.prediction_norm,
                              collections=[self.mode + "/model_summary"])

        self.summary_update = tf.summary.merge_all(self.mode+"/model_summary")

    @classmethod
    def get_model_config(cls, args):
        config = dict()
        config['seed'] = args.seed
        config['model_type'] = args.model_type
        config['data_type'] = args.data_type
        config['use_h36m'] = args.use_h36m

        config['no_normalization'] = args.no_normalization
        config['batch_size'] = args.batch_size
        config['source_seq_len'] = args.seq_length_in
        config['target_seq_len'] = args.seq_length_out
        
        config['early_stopping_tolerance'] = args.early_stopping_tolerance
        config['num_epochs'] = args.num_epochs
        config['print_frequency'] = args.print_frequency
        config['test_frequency'] = args.test_frequency
        
        config['learning_rate'] = args.learning_rate
        config['learning_rate_decay_steps'] = args.learning_rate_decay_steps
        
        config["experiment_id"] = str(int(time.time()))
    
        experiment_name_format = "{}-{}-{}_{}-b{}-in{}_out{}"
        experiment_name = experiment_name_format.format(config["experiment_id"],
                                                        args.model_type,
                                                        "h36m" if args.use_h36m else "amass",
                                                        args.data_type,
                                                        args.batch_size,
                                                        args.seq_length_in,
                                                        args.seq_length_out)
        return config, experiment_name