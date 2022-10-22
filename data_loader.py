import tensorflow as tf
import glob
import os 

def get_train_dataset(batch_size, train_images_paths):
    data_path = os.path.abspath("./data/train/*.png")
    train_images_paths = sorted(glob.glob(data_path))
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images_paths))
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images_paths))
    train_dataset = train_dataset.shuffle(len(train_images_paths))
    train_dataset = train_dataset.map(load_train_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_generator = train_dataset.batch(batch_size)
    
    return train_generator

def load_image(image_path, crop_pad_size=400):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    #   img = augment_image(img)

    size = tf.shape(img)
    
    height, width = size[0], size[1]


    if height < crop_pad_size or width < crop_pad_size:
        pad_h = tf.maximum(0, crop_pad_size - height)
        pad_w = tf.maximum(0, crop_pad_size - width)
                
        # height (top and bottom), width (left and right),   channels
        padding = [[0, pad_h], [0, pad_w], [0, 0]]
        img = tf.pad(img, padding, "REFLECT") 

    size = tf.shape(img)
    height, width = size[0], size[1]

    if height > crop_pad_size or width > crop_pad_size:
        if (height - crop_pad_size) <= 0:
            top = 0
        else:
            top = tf.random.uniform([], 0, height - crop_pad_size, dtype=tf.dtypes.int32)

        if (width - crop_pad_size) <= 0:
            left = 0
        else:
            left = tf.random.uniform([], 0, width - crop_pad_size, dtype=tf.dtypes.int32)

        img = tf.image.crop_to_bounding_box(img, top, left, crop_pad_size, crop_pad_size)

    return img

def load_train_image(image_path):
    img = load_image(image_path)

    return img