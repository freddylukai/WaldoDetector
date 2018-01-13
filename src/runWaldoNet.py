import os
import tensorflow as tf
from src import WaldoNet

_MOMENTUM = 0.9
_LEARNING_RATE = 0.01
_BATCH_SIZE = 32
_ITERATIONS = 10000
_MODEL_DIR = None
_DATA_DIR = None
_DATA_FILE = None

def getFileNames(dir):
    fnames = []
    for file in os.listdir(dir):
        fnames.append(file)
    return fnames

def decodeImage(file, location):
    image = tf.image.decode_png(_DATA_DIR+"/"+file, channels=3)
    return image, tf.one_hot(location[0], location[1])

def train_input_fn(locations):
    dataset = tf.data.Dataset.from_tensor_slices(getFileNames(_DATA_DIR))
    dataset.shuffle(buffer_size=1024)
    dataset = dataset.map(lambda value: decodeImage(value, locations[value]))
    dataset = dataset.prefetch(4*_BATCH_SIZE)
    dataset = dataset.batch(_BATCH_SIZE)
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    return images, labels

def model_fn(features, labels, mode):
    logits = WaldoNet.inference(x=features)
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
    filedict = []
    with open(_DATA_FILE) as f:
        for line in f:
            s = line.split(",")
            filedict[s[0]] = (int(s[1]), int(s[2]))
    run_config = tf.estimator.RunConfig().replace(save_checkpoint_steps=500, save_summary_steps=50, model_dir=_MODEL_DIR)
    waldo_finder = tf.estimator.Estimator(model_fn=model_fn, model_dir=_MODEL_DIR, config=run_config)
    waldo_finder.train(input_fn=lambda: train_input_fn(filedict))

if __name__ == "__main__":
    tf.app.run()

