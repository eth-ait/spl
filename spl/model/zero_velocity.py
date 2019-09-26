import time
import tensorflow as tf

from base_model import BaseModel


class ASimpleYetEffectiveBaseline(BaseModel):
    """Repeats the last known frame for as many frames necessary.
    
    From Martinez et al. (https://arxiv.org/abs/1705.02445).
    """
    def __init__(self, config, data_pl, mode, reuse, **kwargs):
        super(ASimpleYetEffectiveBaseline, self).__init__(config, data_pl, mode, reuse, **kwargs)

        # Dummy variable
        self._dummy = tf.Variable(0.0, name="imadummy")
        
        # Extract the seed and target inputs.
        with tf.name_scope("inputs"):
            self.encoder_inputs = self.data_inputs[:, 0:self.source_seq_len - 1]
            self.decoder_inputs = self.data_inputs[:, self.source_seq_len - 1:-1]
            self.decoder_outputs = self.data_inputs[:, self.source_seq_len:]
    
            enc_in = tf.reshape(tf.transpose(self.encoder_inputs, [1, 0, 2]), [-1, self.input_size])
            dec_in = tf.reshape(tf.transpose(self.decoder_inputs, [1, 0, 2]), [-1, self.input_size])
            dec_out = tf.reshape(tf.transpose(self.decoder_outputs, [1, 0, 2]), [-1, self.input_size])
    
            self.enc_in = tf.split(enc_in, self.source_seq_len - 1, axis=0)
            self.dec_in = tf.split(dec_in, self.target_seq_len, axis=0)
            self.dec_out = tf.split(dec_out, self.target_seq_len, axis=0)
            self.prediction_inputs = self.decoder_inputs
            self.prediction_targets = self.decoder_outputs

    def build_network(self):
        # Don't do anything, just repeat the last known pose.
        last_known_pose = self.decoder_inputs[:, 0:1]
        self.outputs_mu = tf.tile(last_known_pose, [1, self.target_seq_len, 1])
        self.outputs = self.outputs_mu

    def build_loss(self):
        d = self._dummy - self._dummy
        self.loss = tf.reduce_mean(tf.reduce_sum(d*d))

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
    
        config['num_epochs'] = args.num_epochs
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
