from PIL import Image
import numpy as np
import tensorflow as tf
import random

OUTPUT_PATH = r'E:\COS 429\FinalProj\ImagesAsTFRecords'
DATA_DIR = r"E:\COS 429\FinalProj\OriginalJPG"
DICTIONARY_FILE = r'E:\COS 429\FinalProj\stuff.csv'


def create_tf_example(fname, x, y, boxw, boxh, w, h):
  height = h # Image height
  width = w # Image width
  filename = fname # Filename of the image. Empty if image is not from file
  encoded_image_data =  tf.gfile.FastGFile(DATA_DIR+"/"+fname, 'rb').read() # Encoded image bytes
  image_format = b'jpeg' # b'jpeg' or b'png'

  xmins = [(x-boxw+0.0)/w] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [(x+boxw+0.0)/w] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [(y-boxh+0.0)/h] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [(y+boxh+0.0)/h] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [b'Waldo'] # List of string class name of bounding box (1 per box)
  classes = [1] # List of integer class id of bounding box (1 per box)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
      'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
      'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(filename)])),
      'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(filename)])),
      'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image_data])),
      'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
      'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
      'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
      'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
      'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
      'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
      'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
  }))
  return tf_example


def main(_):
  filedict = []
  with open(DICTIONARY_FILE) as f:
      for line in f:
          s = line.split(",")
          im = Image.open(DATA_DIR+"/"+s[0]+'.jpg')
          width, height = im.size
          filedict.append((s[0]+'.jpg', int(s[1]), int(s[2]), width, height, int(s[3])//2, int(s[4])//2))

  fname = 'zrecord-eval.tfrecord'
  writer = tf.python_io.TFRecordWriter(OUTPUT_PATH+'\\'+fname)
  for file, x, y, w, h, boxw, boxh in filedict:
      tf_example = create_tf_example(file, x, y, boxw, boxh, w, h)
      writer.write(tf_example.SerializeToString())
  writer.close()


if __name__ == '__main__':
  tf.app.run()