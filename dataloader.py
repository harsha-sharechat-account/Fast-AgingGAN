from utils import read_image_label_txt, read_image_label_pair_txt
import tensorflow as tf
import random


class DataLoaderAge(object):
    """Data Loader for the age classifier, that prepares a tf data object for training."""

    def __init__(self, image_dir, text_file, image_size):
        """
        Initializes the dataloader.
        Args:
            image_dir: The path to the directory containing high resolution images.
            text_file: Integer, filename and age labe text file.
            image_size: The size of images to train on.
        Returns:
            The dataloader object.
        """
        self.image_paths, self.image_labels = read_image_label_txt(image_dir, text_file)
        self.image_size = image_size

    def _parse_image(self, image_path, image_label):
        """
        Function that loads the images given the path.
        Args:
            image_path: Path to an image file.
            image_label: Integer label of age category.
        Returns:
            image: A tf tensor of the loaded image.
            image_label: A tf tensor of the loaded age label.
        """

        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)

        return image, image_label

    def _resize_image(self, image, label):
        """Resizes the given image
        Args:
            image: tf tensor to resize.
            label: Image class label.
        Returns:
            image: tf tensor of resized image.
            label: Image class label.
        """

        image = tf.image.resize(image, [self.image_size, self.image_size])

        return image, label

    def dataset(self, batch_size):
        """
        Args:
            batch_size: The batch size of the loaded data.
        returns:
            dataset: A tf dataset object.
        """

        dataset = tf.data.Dataset.from_tensor_slices((self.image_paths, self.image_labels))

        # Read the images
        dataset = dataset.map(self._parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Resize the image
        dataset = dataset.map(self._resize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Batch the image
        dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

        return dataset


class DataLoaderGAN(object):
    """Data Loader for the SR GAN, that prepares a tf data object for training."""

    def __init__(self, image_dir, text_dir, image_size):
        """
        Initializes the dataloader.
        Args:
            image_dir: The path to the directory containing high resolution images.
            text_dir: Path to the directory with the all the image list split text files.
            image_size: Size of the images to train on.
        Returns:
            None
        """
        self.label_pairs, self.image_pairs = read_image_label_pair_txt(image_dir, text_dir)
        self.source_images = [x[0] for x in self.image_pairs]
        random.shuffle(self.source_images)
        self.image_size = image_size

    def _parse_image(self, source_path, image_paths, image_labels):
        """
        Function that loads the images given the path.
        Args:
            source_path: The path to the source image.
            image_path: List of path to target and non target-images.
            image_labels: List of targer and non-target image age labels.
        Returns:
            source_image: tf tensor of the source image.
            true_image: tf tensor of the target domain image.
            true_label: tf tensor of true image class label.
            false_label: tf tensor of false image class label.
        """
        source_image = tf.io.read_file(source_path)
        true_image = tf.io.read_file(image_paths[0])

        source_image = tf.image.decode_jpeg(source_image, channels=3)
        true_image = tf.image.decode_jpeg(true_image, channels=3)

        source_image = tf.image.convert_image_dtype(source_image, tf.float32)
        true_image = tf.image.convert_image_dtype(true_image, tf.float32)

        true_label = image_labels[0]
        false_label = image_labels[1]

        return source_image, true_image, true_label, false_label

    def _rescale(self, source_image, true_image, true_label, false_label):
        """
        Function that rescales the pixel values to the -1 to 1 range.
        For use with the generator output tanh function.
        Args:
            source_image: tf tensor of the source image.
            true_image: tf tensor of the target domain image.
            true_label: tf tensor of true image class label.
            false_label: tf tensor of false image class label.
        Returns:
            source_image: tf tensor of the source image, rescaled to -1 to 1 values..
            true_image: tf tensor of the target domain image, rescaled to -1 to 1 values.
            true_label: tf tensor of true image class label.
            false_label: tf tensor of false image class label.
        """
        source_image = source_image * 2.0 - 1.0
        true_image = true_image * 2.0 - 1.0

        return source_image, true_image, true_label, false_label

    def _resize(self, source_image, true_image, true_label, false_label):
        """
        Function that resizes the image.
        Args:
            source_image: tf tensor of the source image.
            true_image: tf tensor of the target domain image.
            true_label: tf tensor of true image class label.
            false_label: tf tensor of false image class label.
        Returns:
            source_image: tf tensor of the source image, resized.
            true_image: tf tensor of the target domain image, resized.
            true_label: tf tensor of true image class label.
            false_label: tf tensor of false image class label.
        """
        source_image = tf.image.resize(source_image, [self.image_size, self.image_size])
        true_image = tf.image.resize(true_image, [self.image_size, self.image_size])

        return source_image, true_image, true_label, false_label

    def _condition(self, source_image, true_image, true_label, false_label):
        """
        Creates image conditioning for aging.
        Args:
            source_image: tf tensor of the source image.
            true_image: tf tensor of the target domain image.
            true_label: tf tensor of true image class label.
            false_label: tf tensor of false image class label.
        Returns:
            source_conditioned_image: The source image, with depthwise concatenated age condition.
            true_image: tf tensor of the target domain image.
            true_condition: tf tensor of the target domain age labels.
            false_condition: tf tensor of non target domain age labels.
            true_label: tf tensor of true image class label.
        """
        true_condition = tf.tile([true_label], [self.image_size * self.image_size])
        true_condition = tf.reshape(true_condition, [self.image_size, self.image_size, 1])
        true_condition = tf.cast(true_condition, tf.float32)

        false_condition = tf.tile([false_label], [self.image_size * self.image_size // 4])
        false_condition = tf.reshape(false_condition, [self.image_size // 2, self.image_size // 2, 1])
        false_condition = tf.cast(false_condition, tf.float32)

        source_conditioned_image = tf.concat([source_image, true_condition], axis=-1)

        true_condition = tf.tile([true_label], [self.image_size * self.image_size // 4])
        true_condition = tf.reshape(true_condition, [self.image_size // 2, self.image_size // 2, 1])
        true_condition = tf.cast(true_condition, tf.float32)

        return source_conditioned_image, true_image, true_condition, false_condition, true_label

    def dataset(self, batch_size):
        """
        Returns a tf dataset object with specified mappings.
        Args:
            batch_size: Int, The number of elements in a batch returned by the dataset.
        Returns:
            dataset: A tf dataset object.
        """
        # Values in range -1 - 1
        # Generate tf dataset from high res image paths.
        dataset = tf.data.Dataset.from_tensor_slices((self.source_images, self.image_pairs, self.label_pairs))

        # Read the images
        dataset = dataset.map(self._parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Resize the image
        dataset = dataset.map(self._resize, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Rescale the values in the input
        dataset = dataset.map(self._rescale, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Condition the age category on the image
        dataset = dataset.map(self._condition, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Batch the input, drop remainder to get a defined batch size.
        # Prefetch the data for optimal GPU utilization.
        dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

        return dataset
