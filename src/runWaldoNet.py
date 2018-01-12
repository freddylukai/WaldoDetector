import tensorflow as tf
from src import WaldoNet

_MOMENTUM = 0.9
_LEARNING_RATE = 0.01
_BATCH_SIZE = 32
_ITERATIONS = 10000
_MODEL_DIR = None
_DATA_DIR = None
_DATA_FILE = None

def train_input_fn():
    pass

def model_fn(images, labels, mode):
    network = WaldoNet.inference
    logits = network(x=images)
    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='prediction_probability')
    }
    _loss = WaldoNet.loss(logits, labels)
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(_LEARNING_RATE, global_step=global_step, decay_steps=5000, decay_rate=0.99, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=_MOMENTUM)
        train_op = optimizer.minimize(_loss, global_step)
    else:
        train_op = None
    accuracy = tf.metrics.accuracy(
        tf.argmax(labels, axis=1), predictions['classes'])
    metrics = {'accuracy': accuracy}
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=_loss,
        train_op=train_op,
        eval_metric_ops=metrics)

def main(unused_argv):
    run_config = tf.estimator.RunConfig().replace(save_checkpoint_steps=500, save_summary_steps=50, model_dir=_MODEL_DIR)

