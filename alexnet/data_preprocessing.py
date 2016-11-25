import glob
from itertools import groupby
from collections import defaultdict
import tensorflow as tf
import numpy
import os

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('phase', 'test', 'option for this is [train, test]')
flags.DEFINE_integer('batch_size', 10, 'the batch size for training')
flags.DEFINE_string('ckpt_dir', '/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/alexnet/model/alex', 'the path to checkpoint')
flags.DEFINE_string('ckpt_dir_test', '/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/alexnet/model/alex.checkpoint', 'the path to checkpoint')
flags.DEFINE_string('logs_dir', '/home/aurora/hdd/workspace/PycharmProjects/sar_edge_detection/alexnet/logs/', 'the path to checkpoint')
flags.DEFINE_float('keep_prob', 0.5, 'keep_prob value ')

# def dense_to_one_hot(labels_dense, num_classes):
#   """Convert class labels from scalars to one-hot vectors."""
#   num_labels = labels_dense.shape[0]
#   index_offset = numpy.arange(num_labels) * num_classes
#   labels_one_hot = numpy.zeros((1, num_classes))
#   labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
#   return labels_one_hot


def dense_to_one_hot2(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  labels_one_hot = numpy.zeros((num_classes), dtype=numpy.float32)
  labels_one_hot[labels_dense] = 1
  return labels_one_hot


def write_records_file(dataset, record_location, sess, labels):
    """
    Fill a TFRecords file with the images found in 'dataset' and include their category.

    Parameters
    :param dataset: dict(list)
        Dictionary with each key being a label for the list of image filenames of its value.
    :param record_location: str
        Location to store the TFRecord output.
    :return:
    """
    writer = None
    current_index = 0
    for breed, images_filenames in dataset.items():
        for image_filename in images_filenames:
            if current_index % 100 == 0:
                if writer:
                    writer.close()
                record_filename = '{record_location}-{current_index}.tfrecords'.format(record_location=record_location,
                                                                                       current_index=current_index)
                writer = tf.python_io.TFRecordWriter(record_filename)
            current_index += 1
            image_file = tf.read_file(image_filename)
            try:
                image = tf.image.decode_jpeg(image_file)
            except:
                print image_filename
                continue
            grayscale_image = tf.image.rgb_to_grayscale(image)
            resized_image = tf.image.resize_images(grayscale_image, 250, 151)
            image_bytes = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()
            labels_one_hot = dense_to_one_hot2(labels.index(breed), 120).tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[labels_one_hot])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
            }))
            writer.write(example.SerializeToString())
    writer.close()


def load_image():
    filenames = tf.train.match_filenames_once("/home/aurora/hdd/workspace/data/imagenet_dog/record_data/training2/*.tfrecords")
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=1)
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized,
        features={
            'label': tf.FixedLenFeature([], tf.string),
            'image': tf.FixedLenFeature([], tf.string)
        })
    record_image = tf.decode_raw(features['image'], tf.uint8)

    image = tf.reshape(record_image, [250, 151, 1])
    record_label = tf.decode_raw(features['label'], tf.float32)
    label = tf.reshape(record_label, [120])
    min_after_dequeue = 100
    batch_size = FLAGS.batch_size
    capacity = min_after_dequeue + 3*batch_size
    image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)
    return image_batch, label_batch


def load_image_test():
    filenames = tf.train.match_filenames_once("/home/aurora/hdd/workspace/data/imagenet_dog/record_data/testing2/*.tfrecords")
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=1)
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized,
        features={
            'label': tf.FixedLenFeature([], tf.string),
            'image': tf.FixedLenFeature([], tf.string)
        })
    record_image = tf.decode_raw(features['image'], tf.uint8)

    image = tf.reshape(record_image, [250, 151, 1])
    record_label = tf.decode_raw(features['label'], tf.float32)
    label = tf.reshape(record_label, [120])
    min_after_dequeue = 10
    batch_size = FLAGS.batch_size
    capacity = min_after_dequeue + 3*batch_size
    image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)
    # sess.run(tf.initialize_all_variables())
    return image_batch, label_batch



def cnn_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def cnn_bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv2d(image, w):
    return tf.nn.conv2d(image, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(image):
    return tf.nn.max_pool(image, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def inference(image_batche):
    float_image_batches = tf.image.convert_image_dtype(image_batche, tf.float32)
    input_image = tf.reshape(float_image_batches, [-1, 250, 151, 1])

    # conv1
    conv1_w = cnn_weights([5, 5, 1, 32])
    conv1_b = cnn_bias([32])
    h_conv1 = tf.nn.relu(conv2d(input_image, conv1_w) + conv1_b)
    h_pool1 = max_pool(h_conv1)

    # conv2
    conv2_w = cnn_weights([5, 5, 32, 64])
    conv2_b = cnn_bias([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, conv2_w)+conv2_b)
    h_pool2 = max_pool(h_conv2)

    # conv3

    # fc1
    h_pool2_flat = tf.reshape(h_pool2, [-1, 63*38*64])
    fc1_w = cnn_weights([63*38*64, 1024])
    fc1_b = cnn_bias([1024])
    fc1_logit = tf.nn.relu(tf.matmul(h_pool2_flat, fc1_w) + fc1_b)
    fc1_output = tf.nn.dropout(fc1_logit, keep_prob=FLAGS.keep_prob)

    # fc2
    fc2_w = cnn_weights([1024, 120])
    fc2_b = cnn_bias([120])
    fc2_logit = tf.nn.relu(tf.matmul(fc1_output, fc2_w) + fc2_b)
    return fc2_logit


def gen_record_files(url):
    labels = list(map(lambda filename: filename.split('/')[-1],
                      glob.glob('/home/aurora/hdd/workspace/data/imagenet_dog/Images/*')))
    image_filenames = glob.glob(url)
    training_dataset = defaultdict(list)
    testing_dataset = defaultdict(list)
    image_filename_with_breed = map(lambda filename: (filename.split("/")[-2], filename), image_filenames)
    for dog_breed, breed_images in groupby(image_filename_with_breed, lambda x: x[0]):
        for i, breed_image in enumerate(breed_images):
            if i % 5 == 0:
                testing_dataset[dog_breed].append(breed_image[1])
            else:
                training_dataset[dog_breed].append(breed_image[1])
        breed_training_count = float(len(training_dataset[dog_breed]))
        breed_testing_count = float(len(testing_dataset[dog_breed]))
        assert round(breed_testing_count/(breed_training_count+breed_testing_count), 2) > 0.18, "Not enough data"
    # write_records_file(training_dataset, '/home/aurora/hdd/workspace/data/imagenet_dog/record_data/traing2/', sess, labels)
    # write_records_file(testing_dataset, '/home/aurora/hdd/workspace/data/imagenet_dog/record_data/testing2', sess, labels)
    #write_records_file(training_dataset, '/home/aurora/hdd/workspace/data/imagenet_dog/record_data/traing2/', sess,labels)
    write_records_file(testing_dataset, '/home/aurora/hdd/workspace/data/imagenet_dog/record_data/test/test', sess, labels)

if __name__ == '__main__':
    url = '/home/aurora/hdd/workspace/data/imagenet_dog/Images/n02*/*.jpg'
    # keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='prob')
    image_batches, label_batches = load_image()
    # image_batches, label_batches = load_image_test()
    logit = inference(image_batches)
    # if FLAGS.phase == 'test':
    #     image_batches = None
    #     label_batches = None
    #     logit = None
    #     image_batches, label_batches = load_image_test()
    #     logit = inference(image_batches)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logit, label_batches))
    tf.scalar_summary('cross_entropy', loss)
    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
    sess = tf.Session()
    # graph writer
    merged_summary = tf.merge_all_summaries()
    pretrain_writer = tf.train.SummaryWriter(FLAGS.logs_dir, sess.graph)
    # saver
    saver = tf.train.Saver()
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    count = 0
    real_accuracy = 0
    sum_accuracy = 0
    try:
        if FLAGS.phase == 'train':
            while not coord.should_stop():
                count += 1
                # Run training steps or whatever
                sess.run(train_op)
                if count%100 == 0:
                    real_loss, merged = sess.run([loss, merged_summary])
                    print("Step %d, loss = %f" % (count, real_loss))
                    pretrain_writer.add_summary(merged,  global_step=count)
        elif FLAGS.phase == 'test':
            count = 0
            accuracy = 0
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(FLAGS.ckpt_dir_test))
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                while not coord.should_stop():
                    count += 1
                    logit_val = sess.run(logit)
                    # accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(logit, 1), tf.argmax(label_batches, 1)), tf.float32))
                    # real_accuracy = sess.run(accuracy)
                    # sum_accuracy = sum_accuracy + real_accuracy
                    # print 'batch accuracy: ', real_accuracy, 'sum_accuracy is: ', sum_accuracy
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
        if FLAGS.phase == 'train':
            saver.save(sess, FLAGS.ckpt_dir)
            print count
        elif FLAGS.phase == 'test':
            print 'test accuracy is ', sum_accuracy, count, sum_accuracy/count*FLAGS.batch_size
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
