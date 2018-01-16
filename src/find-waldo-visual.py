from matplotlib import pyplot as plt
import numpy as np
import sys
import tensorflow as tf
import matplotlib
from PIL import Image
import matplotlib.patches as patches

model_path = r'E:\COS 429\FinalProj\FinalGraph\frozen_inference_graph.pb'
image_path = r'E:\COS 429\FinalProj\OriginalJPG\Beach.png.jpg'


def draw_box(box, image_np):
    # expand the box by 50%
    box += np.array([-(box[2] - box[0]) / 2, -(box[3] - box[1]) / 2, (box[2] - box[0]) / 2, (box[3] - box[1]) / 2])

    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)

    # draw blurred boxes around box
    ax.add_patch(patches.Rectangle((0, 0), box[1] * image_np.shape[1], image_np.shape[0], linewidth=0, edgecolor='none',
                                   facecolor='w', alpha=0.8))
    ax.add_patch(patches.Rectangle((box[3] * image_np.shape[1], 0), image_np.shape[1], image_np.shape[0], linewidth=0,
                                   edgecolor='none', facecolor='w', alpha=0.8))
    ax.add_patch(patches.Rectangle((box[1] * image_np.shape[1], 0), (box[3] - box[1]) * image_np.shape[1],
                                   box[0] * image_np.shape[0], linewidth=0, edgecolor='none', facecolor='w', alpha=0.8))
    ax.add_patch(patches.Rectangle((box[1] * image_np.shape[1], box[2] * image_np.shape[0]),
                                   (box[3] - box[1]) * image_np.shape[1], image_np.shape[0], linewidth=0,
                                   edgecolor='none', facecolor='w', alpha=0.8))

    return fig, ax


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(model_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

def closest_500(i):
    return (i//500)*500 if i%500 < 250 else ((i//500)+1)*500

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    size = closest_500(im_width), closest_500(im_height)
    newimg = image.resize(size)
    return np.array(newimg.getdata()).reshape(
        (size[1], size[0], 3)).astype(np.uint8)


with detection_graph.as_default():
    detections = []
    subimages = []
    with tf.Session(graph=detection_graph) as sess:
        image_np = load_image_into_numpy_array(Image.open(image_path))
        # Actual detection.
        shape = image_np.shape
        for i in range(0, shape[0]//500):
            for j in range(0, shape[1]//500):
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                subimg = image_np[i*500:(i+1)*500, j*500:(j+1)*500, :]
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: np.expand_dims(subimg, axis=0)})
                if scores[0][0] > 0.99:
                    detections.append(boxes[0][0])
                    subimages.append(subimg)


        if len(detections) == 0:
            print("Waldo not found :(")
        else:
            for i, subimg in enumerate(subimages):
                fig, ax = draw_box(detections[i], subimg)
                ax.imshow(subimg)
                plt.show()
