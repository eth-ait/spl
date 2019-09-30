import time
import tensorflow as tf

from model.base_model import BaseModel


class ASimpleYetEffectiveBaseline(BaseModel):
    """Repeats the last known frame for as many frames necessary.
    
    From Martinez et al. (https://arxiv.org/abs/1705.02445).
    """
    def __init__(self, config, data_pl, mode, reuse, **kwargs):
        super(ASimpleYetEffectiveBaseline, self).__init__(config, data_pl, mode, reuse, **kwargs)

        # Dummy variable
        self._dummy = tf.Variable(0.0, name="dummy_variable")
        
        # Extract the seed and target inputs.
        with tf.name_scope("inputs"):
            self.prediction_inputs = self.data_inputs[:, self.source_seq_len - 1:-1]
            self.prediction_targets = self.data_inputs[:, self.source_seq_len:]

    def build_network(self):
        # Don't do anything, just repeat the last known pose.
        last_known_pose = self.prediction_inputs[:, 0:1]
        self.outputs = tf.tile(last_known_pose, [1, self.target_seq_len, 1])

    def build_loss(self):
        # Build a loss operation so that training script doesn't complain.
        d = self._dummy - self._dummy
        self.loss = tf.reduce_mean(tf.reduce_sum(d*d))
        
    def summary_routines(self):
        # Build a summary operation so that training script doesn't complain.
        tf.summary.scalar(self.mode+"/loss", self.loss, collections=[self.mode+"/model_summary"])
        self.summary_update = tf.summary.merge_all(self.mode + "/model_summary")
    
    def step(self, session):
        output_feed = [self.loss,
                       self.summary_update,
                       self.outputs]
        outputs = session.run(output_feed)
        return outputs[0], outputs[1], outputs[2]
    
    def sampled_step(self, session):
        assert self.is_eval, "Only works in sampling mode."
        prediction, targets, seed_sequence, data_id = session.run([self.outputs,
                                                                   self.prediction_targets,
                                                                   self.data_inputs[:, :self.source_seq_len],
                                                                   self.data_ids])
        return prediction, targets, seed_sequence, data_id

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
    
        config['num_epochs'] = 0
        config['print_frequency'] = args.print_frequency
        config['test_frequency'] = args.test_frequency
    
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