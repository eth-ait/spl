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
        
        self.use_quat = config['data_type'] == C.QUATERNION
        self.use_aa = config['data_type'] == C.ANGLE_AXIS
        self.use_rotmat = config['data_type'] == C.ROT_MATRIX
        self.use_h36m = config.get("use_h36m", False)

        # Set by the child model classes.
        self.outputs = None  # List of predicted frames. If the model is probabilistic, a sample is drawn first.
        self.prediction_targets = None  # Targets in pose loss term.
        self.prediction_inputs = None  # Inputs that are used to make predictions.
        # Intermediate representation of the model to make predictions. It can be output of the RNN cell which is later
        # fed to a dense output layer.
        self.prediction_representation = None
        
        # Training
        self.loss = None  # Loss op to be used in training.
        self.gradient_norms = None  #
        self.parameter_update = None  # Parameter update op: optimizer output.
        self.summary_update = None  # Summary op to write summaries.
        self.learning_rate = None

        # Hard-coded parameters.
        self.JOINT_SIZE = 4 if self.use_quat else 3 if self.use_aa else 9
        self.NUM_JOINTS = 21 if self.use_h36m else 15
        self.HUMAN_SIZE = self.NUM_JOINTS*self.JOINT_SIZE
        self.input_size = self.HUMAN_SIZE

    def build_graph(self):
        """Creates Tensorflow training graph by building the actual network and calculating the loss operation."""
        self.build_network()
        self.build_loss()

    def build_network(self):
        """Builds the network taking the inputs and yielding the model output."""
        pass

    def build_loss(self):
        """Calculates the loss terms to train the network."""
        pass

    def optimization_routines(self):
        """Creates and optimizer, applies gradient regularizations and sets the parameter_update operation."""
        pass

    def step(self, session):
        """Runs one training step by evaluating loss, parameter update, summary and output operations.
        
        Model receives data from the data pipeline automatically. In contrast to `sampled_step`, model's output is not
        fed back to the model.
        Args:
            session: TF session object.
        Returns:
            loss, summary proto, prediction
        """
        pass

    def sampled_step(self, session):
        """Runs an auto-regressive sampling step. It is used to evaluate the model.
        
        In contrast to `step`, predicted output step is fed back to the model to predict the next step.
        Args:
            session: TF session object.
        Returns:
            predicted sequence, actual target, input sequence, sample's data_id
        """
        pass

    def summary_routines(self):
        """Creates Tensorboard summaries."""
        pass

    @classmethod
    def get_model_config(cls, args):
        """Given command-line arguments, creates the configuration dictionary.
        
        It is later passed to the models and stored in the disk.
        Args:
            args: command-line argument object.
        Returns:
            experiment configuration (dict), experiment name (str)
        """
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
