import tensorflow  as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.estimator.model_fn import EstimatorSpec, ModeKeys
from attention.modules import TransformerModule

from attention.algorithms.transformer.inputs_fn import get_input_fn

tf.logging.set_verbosity(tf.logging.INFO)


class TransformerAlgorithm:
    def __init__(self, estimator_run_config, params=None):
        self.model_params = params
        self.estimator = tf.estimator.Estimator(self.get_model_fn(),
                                                params=self.model_params,
                                                config=estimator_run_config,
                                                model_dir=estimator_run_config.model_dir)
        self.experiment = None
        self.training_params = {}
        self.pad_token = self.model_params.get("pad_token", 0)

    def get_model_fn(self):
        def model_fn(features, labels, mode, params=None, config=None):
            train_op = None
            mean_loss = None
            eval_metrics = None
            predictions = None
            step = None
            transformer_model = TransformerModule(params=self.model_params, mode_key=mode)
            output = transformer_model(features)


            if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
                decoder_inputs = features[1][0]

                step = slim.get_or_create_global_step()
                max_sequence_length = tf.shape(features[1][0])[1]
                output_sequence_len = features[1][1]
                one_hot_labels = tf.one_hot(decoder_inputs[:, 1:], self.model_params.vocab_size, axis=-1)
                output = tf.Print(output, [tf.argmax(output[0], dimension=-1), decoder_inputs[0, 1:]])
                with tf.name_scope("loss"):
                    mask_loss = tf.sequence_mask(output_sequence_len, maxlen=max_sequence_length - 1, dtype=tf.float32)
                    one_hot_labels = tf.reshape(one_hot_labels, [-1, self.model_params.vocab_size])
                    loss = tf.nn.softmax_cross_entropy_with_logits(
                        logits=tf.reshape(output, [-1, self.model_params.vocab_size]),
                        labels=one_hot_labels)
                    loss = tf.reshape(loss, [-1, max_sequence_length - 1])
                    loss *= mask_loss
                    loss = tf.reduce_sum(loss, 1) / tf.reduce_sum(mask_loss, 1)
                    mean_loss = tf.reduce_mean(loss)

            if mode == ModeKeys.TRAIN:
                train_op = slim.optimize_loss(loss=mean_loss,
                                              global_step=step,
                                              learning_rate=self.training_params["learning_rate"],
                                              clip_gradients=self.training_params["clip_gradients"],
                                              optimizer=params["optimizer"],
                                              summaries=slim.OPTIMIZER_SUMMARIES
                                              )

            if mode == ModeKeys.EVAL:
                eval_metrics = {"loss", mean_loss}

            if mode == ModeKeys.PREDICT:
                predictions = output

            return EstimatorSpec(train_op=train_op, loss=mean_loss, eval_metric_ops=eval_metrics,
                                 predictions=predictions,
                                 mode=mode)

        return model_fn

    def train(self, train_params, train_context_filename, train_answer_filename, extra_hooks=None):
        self.training_params = train_params

        input_fn = get_input_fn(batch_size=train_params["batch_size"], num_epochs=train_params["num_epochs"],
                                context_filename=train_context_filename,
                                answer_filename=train_answer_filename)

        hooks = extra_hooks or []
        self.estimator.train(input_fn=input_fn, steps=train_params.get("steps", None),
                             max_steps=train_params.get("max_steps", None), hooks=hooks)

    def train_and_evaluate(self, train_params, train_context_filename, train_answer_filename, validation_params,
                           validation_context_filename, validation_answer_filename, extra_hooks=None):
        self.training_params = train_params

        input_fn = get_input_fn(batch_size=train_params["batch_size"],
                                num_epochs=train_params["num_epochs"],
                                context_filename=train_context_filename,
                                answer_filename=train_answer_filename,
                                max_sequence_len=train_params["max_sequence_len"])

        validation_input_fn = get_input_fn(batch_size=validation_params["batch_size"],
                                           num_epochs=validation_params["num_epochs"],
                                           context_filename=validation_context_filename,
                                           answer_filename=validation_answer_filename,
                                           max_sequence_len=validation_params["max_sequence_len"])

        hooks = extra_hooks or None

        self.experiment = tf.contrib.learn.Experiment(estimator=self.estimator,
                                                      train_input_fn=input_fn,
                                                      eval_input_fn=validation_input_fn,
                                                      train_steps=train_params.get("steps", None),
                                                      eval_steps=validation_params["steps"],
                                                      train_monitors=hooks,
                                                      min_eval_frequency=validation_params.get("min_eval_frequency",
                                                                                               None))

        self.experiment.train()

    def inferrence(self, train_params, context_filename):
        input_fn = get_input_fn(batch_size=1,
                                num_epochs=train_params["num_epochs"],
                                context_filename=context_filename,
                                max_sequence_len=train_params["max_sequence_len"])

        return self.estimator.predict(input_fn=input_fn)
