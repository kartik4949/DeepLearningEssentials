import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

AUTOTUNE = tf.data.experimental.AUTOTUNE


class Flip:
    def __init__(
        self,
        dim="horizontal",
        labels_format={"xmin": 0, "ymin": 1, "xmax": 2, "ymax": 3},
    ):

        if not (dim in {"horizontal", "vertical"}):
            raise ValueError("`dim` can be one of 'horizontal' and 'vertical'.")
        self.dim = dim
        self.labels_format = labels_format

    def __call__(self, image, labels=None, return_inverter=False):

        img_height, img_width = image.shape[:2]

        xmin = self.labels_format["xmin"]
        ymin = self.labels_format["ymin"]
        xmax = self.labels_format["xmax"]
        ymax = self.labels_format["ymax"]

        if self.dim == "horizontal":
            # image = image[:, ::-1]
            image = tf.image.flip_left_right(image).numpy()
            if labels is None:
                return image
            else:
                labels = np.copy(labels)
                labels[:, [xmin, xmax]] = img_width - labels[:, [xmax, xmin]]
                return image, labels
        else:
            # image = image[::-1]
            image = tf.image.flip_left_right(image).numpy()
            if labels is None:
                return image
            else:
                labels = np.copy(labels)
                labels[:, [ymin, ymax]] = img_height - labels[:, [ymax, ymin]]
                return image, labels


class RandomFlip:
    def __init__(
        self,
        dim="horizontal",
        prob=0.5,
        labels_format={"xmin": 0, "ymin": 1, "xmax": 2, "ymax": 3},
    ):

        self.dim = dim
        self.prob = prob
        self.labels_format = labels_format
        self.flip = Flip(dim=self.dim, labels_format=self.labels_format)

    def __call__(self, image, labels=None):
        p = np.random.uniform(0, 1)
        if p >= (1.0 - self.prob):
            self.flip.labels_format = self.labels_format
            return self.flip(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels


class RandomTranslate:
    def __init__(
        self,
        dy_minmax=(0.03, 0.3),
        dx_minmax=(0.03, 0.3),
        prob=0.5,
        clip_boxes=True,
        box_filter=None,
        image_validator=None,
        n_trials_max=3,
        background=(0, 0, 0),
        labels_format={"class_id": 0, "xmin": 1, "ymin": 2, "xmax": 3, "ymax": 4},
    ):

        if dy_minmax[0] > dy_minmax[1]:
            raise ValueError("It must be `dy_minmax[0] <= dy_minmax[1]`.")
        if dx_minmax[0] > dx_minmax[1]:
            raise ValueError("It must be `dx_minmax[0] <= dx_minmax[1]`.")
        if dy_minmax[0] < 0 or dx_minmax[0] < 0:
            raise ValueError("It must be `dy_minmax[0] >= 0` and `dx_minmax[0] >= 0`.")
        self.dy_minmax = dy_minmax
        self.dx_minmax = dx_minmax
        self.prob = prob
        self.clip_boxes = clip_boxes
        self.box_filter = box_filter
        self.image_validator = image_validator
        self.n_trials_max = n_trials_max
        self.background = background
        self.labels_format = labels_format
        self.translate = Translate(
            dy=0,
            dx=0,
            clip_boxes=self.clip_boxes,
            box_filter=self.box_filter,
            background=self.background,
            labels_format=self.labels_format,
        )

    def __call__(self, image, labels):
        p = np.random.uniform(0, 1)
        if p >= (1.0 - self.prob):
            img_height, img_width = image.shape[:2]
            xmin = 0
            ymin = 1
            xmax = 2
            ymax = 3
            for _ in range(max(1, self.n_trials_max)):
                dy_abs = np.random.uniform(self.dy_minmax[0], self.dy_minmax[1])
                dx_abs = np.random.uniform(self.dx_minmax[0], self.dx_minmax[1])
                dy = np.random.choice([-dy_abs, dy_abs])
                dx = np.random.choice([-dx_abs, dx_abs])
                self.translate.dy_rel = dy
                self.translate.dx_rel = dx
                new_labels = np.copy(labels)
                new_labels[:, [ymin, ymax]] += int(round(img_height * dy))
                new_labels[:, [xmin, xmax]] += int(round(img_width * dx))
                for label_new in new_labels:
                    if (
                        (label_new[0] < img_width + 20 and label_new[0] > -20)
                        and (label_new[2] < img_width + 20 and label_new[2] > -20)
                        and (label_new[1] < img_height + 20 and label_new[1] > -20)
                        and (label_new[3] < img_height + 20 and label_new[3] > -20)
                    ):
                        image, labels = self.translate(image, labels)

        return image, labels


class Translate:
    def __init__(
        self,
        dy,
        dx,
        clip_boxes=True,
        box_filter=None,
        background=(0, 0, 0),
        labels_format={"class_id": 0, "xmin": 1, "ymin": 2, "xmax": 3, "ymax": 4},
    ):
        self.dy_rel = dy
        self.dx_rel = dx
        self.clip_boxes = clip_boxes
        self.box_filter = box_filter
        self.background = background
        self.labels_format = labels_format

    def __call__(self, image, labels):
        img_height, img_width = image.shape[:2]

        dy_abs = int(round(img_height * self.dy_rel))
        dx_abs = int(round(img_width * self.dx_rel))
        M = np.float32([[1, 0, dx_abs], [0, 1, dy_abs]])

        image = cv2.warpAffine(
            image,
            M=M,
            dsize=(img_width, img_height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.background,
        )
        xmin = 0
        ymin = 1
        xmax = 2
        ymax = 3

        labels = np.copy(labels)
        labels[:, [xmin, xmax]] += dx_abs
        labels[:, [ymin, ymax]] += dy_abs
        if self.clip_boxes:
            labels[:, [ymin, ymax]] = np.clip(
                labels[:, [ymin, ymax]], a_min=0, a_max=img_height - 1
            )
            labels[:, [xmin, xmax]] = np.clip(
                labels[:, [xmin, xmax]], a_min=0, a_max=img_width - 1
            )
        return image, labels


class Image_Augmentation:
    def __init__(self, config, in_dir):
        self.path_csv = config["path_csv"]
        self.Dataset_Size = 0
        self.batch_size = config["batch_size"]
        self.Randomscale = RandomTranslate()
        self.Randomflip = RandomFlip()

    def tf_adjust_brightness(self, tf_img, delta=0.2):
        return tf.image.adjust_brightness(tf_img, delta=delta)

    def random_brightness(self, tf_img, max_delta=0.4):
        return tf.image.random_brightness(tf_img, max_delta=max_delta)

    def random_contrast(self, tf_img):
        return tf.image.random_contrast(tf_img, 0.7, 1.3)

    def random_hue(self, tf_img):
        return tf.image.random_hue(tf_img, 0.08)

    def random_saturation(self, tf_img):
        return tf.image.random_saturation(tf_img, 0.6, 1.6)

    def rgb_hsv(self, tf_img):
        return tf.image.rgb_to_hsv(tf_img)

    def hsv_rgb(self, tf_img):
        return tf.image.hsv_to_rgb(tf_img)

    def convert_float32(self, tf_img):
        return tf.image.convert_image_dtype(tf_img, tf.float32)

    def convert_uint8(self, tf_img):
        return tf.image.convert_image_dtype(tf_img, tf.uint8)

    def tf_flip(self, tf_img, labels, flip_type="horizontal"):
        if flip_type == "horizontal":
            tf_img = tf.image.flip_left_right(tf_img)
            labels = labels.numpy()
            print(labels[:, [0, 2]])
            labels[:, [0, 2]] = tf_img.shape[1] - labels[:, [0, 2]]
        return tf_img, labels

    def prepare_for_training(self, ds, cache=True, shuffle_buffer_size=1000):
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def Transformation_Photometric(self, image, label, sequence=1):
        if sequence == 1:
            # image = self.tf_adjust_brightness(image)
            image = self.random_brightness(image)
            image = self.random_contrast(image)
            image = self.random_saturation(image)
        return image, label

    def Transformation_Geometric(self, image, label, sequence=1):
        image, label = self.Randomflip(image, label)
        image, label = self.Randomscale(image, label)
        return image, label

    def Generate(self):
        df = pd.read_csv(self.path_csv)
        images = df.iloc[:, 0]
        classid = df.iloc[:, -1]
        coord = df.iloc[:, 1:-1]
        tf_csv = tf.data.Dataset.from_tensor_slices(
            (images.values, coord.values, classid.values)
        )

        def read(batch_image, batch_coord, batch_class):
            image_path = tf.io.read_file(batch_image)
            batch_image = tf.io.decode_jpeg(image_path)
            return batch_image, batch_coord, batch_class

        tf_csv_stream = tf_csv.map(read, num_parallel_calls=AUTOTUNE)

        train = self.prepare_for_training(tf_csv_stream)
        
        train = train.map(lambda x, y, z: self.Transformation_Photometric(x, y))

        while True:
            batch_images, batch_labels = next(iter(train))
            batch = []
            for i in range(self.batch_size):
                batch_image, batch_label = self.Transformation_Geometric(
                    batch_images[i].numpy(), [batch_labels[i].numpy()]
                )
                batch.append((batch_image, batch_label))
            yield batch


config = {"path_csv": "val.csv", "batch_size": 4}
in_dir = "./"

ia = Image_Augmentation(config, in_dir)
for batch in ia.Generate():
    img = batch[0][0]
    label = batch[0][1][0]
    img = cv2.rectangle(img, (label[0], label[1]), (label[2], label[3]), (255, 0, 0), 2)
    plt.imshow(img)
    plt.show()
