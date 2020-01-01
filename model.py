import keras
import tensorflow as tf


class AgingGAN(object):
    """Aging GAN for faces."""

    def __init__(self, args, age_train=False):
        """
        Initializes the Fast AgingGAN class.
        Args:
            args: CLI arguments that dictate how to build the model.
            age_train: Whether to train the age classifer or use an existing one.
        Returns:
            None
        """
        self.img_dim = args.img_size
        self.img_size = (args.img_size, args.img_size, 3)
        self.iterations = 0

        # Number of inverted residual blocks in the generator
        self.n_residual_blocks = 6

        # Define Optimizers
        self.gen_optimizer = keras.optimizers.Adam(args.lr)
        self.disc_optimizer = keras.optimizers.Adam(args.lr)
        self.cls_optimizer = keras.optimizers.Adam(args.lr)

        # Calculate output shape of D (PatchGAN)
        patch = int(args.img_size / 2 ** 5)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 24  # Realtime Image Enhancement GAN Galteri et al.
        self.df = 64

        # If training age classifier, load only that into memory
        if age_train:
            self.age_classifier = self.build_age_classifier(args.num_classes)
        else:
            # Otherwise load the GAN setup
            self.age_classifier = keras.models.load_model('models/age_classifier.h5')
            self.age_classifier.trainable = False

            # Build and compile the discriminator
            self.discriminator = self.build_discriminator()

            # Build and compile the generator
            self.generator = self.build_generator()

    @tf.function
    def content_loss(self, real, fake, age_labels):
        """
        The content loss for the generator for face aging.
        Args:
            real: The target domain image
            fake: The generated target domain image.
            age_labels: The age class labels for the classifier loss.
        Returns:
            loss: tf tensor of the sum of feature MSE and age classifier loss.
        """
        fake = (fake + 1.0) / 2.0
        real = (real + 1.0) / 2.0
        fake_labels, fake_features = self.age_classifier(fake)
        _, real_features = self.age_classifier(real)
        feature_loss = tf.keras.losses.MeanSquaredError()(real_features, fake_features)
        age_loss = tf.keras.losses.SparseCategoricalCrossentropy()(age_labels, fake_labels)
        return feature_loss + age_loss

    def build_age_classifier(self, num_classes):
        """
        Builds a pre-trained VGG network for image classification
        Args:
            num_classes: The number of classes for the classifier.
        Returns:
            model: A tf keras model for the classifier.
        """
        # Input image to extract features from
        inputs = keras.Input((self.img_dim, self.img_dim, 3))
        features = keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=self.img_size)(inputs)
        x = keras.layers.Flatten()(features)
        x = keras.layers.Dense(512, activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(512, activation='relu')(x)
        x = keras.layers.Dense(num_classes, activation='softmax')(x)

        # Compile the model
        model = keras.models.Model(inputs, [x, features])

        return model

    def build_generator(self):
        """Build the generator that will do the Face Aging task."""

        def _make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            # Make sure that round down does not go down by more than 10%.
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        def residual_block(inputs, filters, block_id, expansion=6, stride=1, alpha=1.0):
            """Inverted Residual block that uses depth wise convolutions for parameter efficiency.
            Args:
                inputs: The input feature map.
                filters: Number of filters in each convolution in the block.
                block_id: An integer specifier for the id of the block in the graph.
                expansion: Channel expansion factor.
                stride: The stride of the convolution.
                alpha: Depth expansion factor.
            Returns:
                x: The output of the inverted residual block.
            """
            channel_axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1

            in_channels = keras.backend.int_shape(inputs)[channel_axis]
            pointwise_conv_filters = int(filters * alpha)
            pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
            x = inputs
            prefix = 'block_{}_'.format(block_id)

            if block_id:
                # Expand
                x = keras.layers.Conv2D(expansion * in_channels,
                                        kernel_size=1,
                                        padding='same',
                                        use_bias=True,
                                        activation=None,
                                        name=prefix + 'expand')(x)
                x = keras.layers.BatchNormalization(axis=channel_axis,
                                                    epsilon=1e-3,
                                                    momentum=0.999,
                                                    name=prefix + 'expand_BN')(x)
                x = keras.layers.Activation('relu', name=prefix + 'expand_relu')(x)
            else:
                prefix = 'expanded_conv_'

            # Depthwise
            x = keras.layers.DepthwiseConv2D(kernel_size=3,
                                             strides=stride,
                                             activation=None,
                                             use_bias=True,
                                             padding='same' if stride == 1 else 'valid',
                                             name=prefix + 'depthwise')(x)
            x = keras.layers.BatchNormalization(axis=channel_axis,
                                                epsilon=1e-3,
                                                momentum=0.999,
                                                name=prefix + 'depthwise_BN')(x)

            x = keras.layers.Activation('relu', name=prefix + 'depthwise_relu')(x)

            # Project
            x = keras.layers.Conv2D(pointwise_filters,
                                    kernel_size=1,
                                    padding='same',
                                    use_bias=True,
                                    activation=None,
                                    name=prefix + 'project')(x)
            x = keras.layers.BatchNormalization(axis=channel_axis,
                                                epsilon=1e-3,
                                                momentum=0.999,
                                                name=prefix + 'project_BN')(x)

            if in_channels == pointwise_filters and stride == 1:
                return keras.layers.Add(name=prefix + 'add')([inputs, x])
            return x

        def deconv2d(layer_input, filters):
            """Upsampling layer to increase height and width of the input.
            Uses PixelShuffle for upsampling.
            Args:
                layer_input: The input tensor to upsample.
                filters: Numbers of expansion filters.
            Returns:
                u: Upsampled input by a factor of 2.
            """
            u = keras.layers.Upsampling2D(size=(2, 2))(layer_input)
            u = keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')(u)
            u = keras.layers.LeakyReLU()(u)
            return u

        # Original image input
        img_lr = keras.Input(shape=(self.img_dim, self.img_dim, 4))

        # Pre-residual block
        x = keras.layers.Conv2D(self.gf, kernel_size=3, strides=1, padding='same')(img_lr)
        x = keras.layers.BatchNormalization()(x)
        c1 = keras.layers.LeakyReLU()(x)

        # Downsample
        x = keras.layers.Conv2D(self.gf, kernel_size=3, strides=2, padding='same')(c1)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)

        # Downsample
        x = keras.layers.Conv2D(self.gf, kernel_size=3, strides=2, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)

        # Propogate through residual blocks
        for idx in range(0, self.n_residual_blocks):
            x = residual_block(x, self.gf, idx)

        # Upsampling
        x = deconv2d(x, self.gf)
        x = deconv2d(x, self.gf)

        # Add face image (only learn the aging features in the residuals)
        x = keras.layers.Add()([x, c1])

        # Generate output
        gen_hr = keras.layers.Conv2D(3, kernel_size=3, strides=1, padding='same', activation='tanh')(x)

        return keras.models.Model(img_lr, gen_hr)

    def build_discriminator(self):
        """Builds a discriminator network based on the Patch-GAN design."""

        def d_block(layer_input, filters, strides=1, bn=True, act=True):
            """Discriminator layer block.
            Args:
                layer_input: Input feature map for the convolutional block.
                filters: Number of filters in the convolution.
                strides: The stride of the convolution.
                bn: Whether to use batch norm or not.
            """
            d = keras.layers.Conv2D(filters, kernel_size=4, strides=strides, padding='same')(layer_input)
            if bn:
                d = keras.layers.BatchNormalization(momentum=0.8)(d)
            if act:
                d = keras.layers.LeakyReLU(alpha=0.2)(d)

            return d

        # Input img
        d0 = keras.layers.Input(shape=self.img_size)
        # Input input condition
        cond = keras.layers.Input(shape=(self.img_dim // 2, self.img_dim // 2, 1))

        d1 = d_block(d0, self.df, strides=2, bn=False)
        d1 = keras.layers.Concatenate()([d1, cond])
        d2 = d_block(d1, self.df * 2, strides=2)
        d3 = d_block(d2, self.df * 4, strides=2)
        d4 = d_block(d3, self.df * 4, strides=2)
        d5 = d_block(d4, 1, strides=2, bn=False, act=False)

        return keras.models.Model([d0, cond], d5)
