from PIL import Image
import numpy as np
import tensorflow as tf
import random

OUTPUT_PATH = r'E:\COS 429\FinalProj\ImagesAsTFRecords2'
DATA_DIR = r"E:\COS 429\FinalProj\TheNewImsJPG"
DATA_FILE = r'E:\COS 429\FinalProj\imagesForTraining.txt'
OTHER_FILE = r'E:\COS 429\FinalProj\stuff.csv'


def create_tf_example(fname, location, w, h):
  height = 500 # Image height
  width = 500 # Image width
  filename = fname # Filename of the image. Empty if image is not from file
  encoded_image_data =  tf.gfile.FastGFile(DATA_DIR+"/"+fname, 'rb').read() # Encoded image bytes
  image_format = b'jpeg' # b'jpeg' or b'png'

  x, y = location//500, location%500
  xmins = [(x-w)/500.0] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [(x+w)/500.0] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [(y-h)/500.0] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [(y+h)/500.0] # List of normalized bottom y coordinates in bounding box
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
  sizedict = []
  with open(OTHER_FILE) as f:
      for line in f:
          s = line.split(',')
          sizedict.append((int(s[3]), int(s[4])))

  filedict = []
  with open(DATA_FILE) as f:
      for line in f:
          s = line.split(",")
          imgnum = int(s[0][-10:-4])
          width, height = sizedict[(imgnum%9125)//125]
          filedict.append((s[0]+'.jpg', int(s[1]), width//2, height//2))

  random.shuffle(filedict)

  for i in range(7):
    fname = 'zrecord-%02d.tfrecord' %(i)
    writer = tf.python_io.TFRecordWriter(OUTPUT_PATH+'\\'+fname)
    for file, location, w, h in filedict[i*10000:(i+1)*10000]:
        tf_example = create_tf_example(file, location, w, h)
        writer.write(tf_example.SerializeToString())
    writer.close()


if __name__ == '__main__':
  tf.app.run()