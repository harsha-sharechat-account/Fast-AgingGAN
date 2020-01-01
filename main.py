from dataloader import DataLoaderAge, DataLoaderGAN
from argparse import ArgumentParser
from model import AgingGAN
import tensorflow as tf
import os

parser = ArgumentParser()
parser.add_argument('--image_dir', type=str,
                    help='Path to face image directory.')
parser.add_argument('--text_dir', default='data_split', type=str,
                    help='Path to face image directory.')
parser.add_argument('--batch_size', default=24, type=int, help='Batch size for training.')
parser.add_argument('--epochs', default=50, type=int, help='Number of epochs for training')
parser.add_argument('--img_size', default=128, type=int, help='Face image input size.')
parser.add_argument('--num_classes', default=5, type=int, help='Number of age categories')
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate for optimizers.')
parser.add_argument('--save_iter', default=200, type=int,
                    help='The number of iterations to save the tensorboard summaries and models.')


@tf.function
def train_step(model, source_img, true_img, true_condition, false_condition, true_label):
    """Single train step function for the AgingGAN.
    Args:
        model: An object that contains a tf keras compiled discriminator model.
        source_img: Face images to to age.
        true_img: The face images in the target domain.
        true_condition: The target age condition.
        false_condition: The non-target age condition.
        true_label: The class label of the target domain.
    Returns:
        d_loss: The mean loss of the discriminator.
    """
    valid = tf.ones((source_img.shape[0],) + model.disc_patch) - tf.random.uniform(
        (source_img.shape[0],) + model.disc_patch) * 0.2
    fake = tf.ones((source_img.shape[0],) + model.disc_patch) * tf.random.uniform(
        (source_img.shape[0],) + model.disc_patch) * 0.2

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # From input image generate older age version version
        generated_img = model.generator(source_img)

        # Train the discriminators (original images = real / generated = Fake)
        valid_prediction = model.discriminator([true_img, true_condition])
        # Train the discriminators (original images = real / generated = Fake)
        false_prediction = model.discriminator([true_img, false_condition])
        # Train the discriminator on predicted image
        fake_prediction = model.discriminator([generated_img, true_condition])

        # Generator loss
        content_loss = model.content_loss(source_img[..., :3], generated_img, true_label)
        adv_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(valid, fake_prediction)
        perceptual_loss = content_loss + adv_loss

        # Discriminator loss
        valid_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(valid, valid_prediction)
        false_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(fake, false_prediction)
        fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(fake, fake_prediction)

        # Avergae out the loss
        d_loss = 0.5 * (valid_loss + 0.5 * (false_loss + fake_loss))

    # Backprop on Generator
    gen_grads = gen_tape.gradient(perceptual_loss, model.generator.trainable_variables)
    model.gen_optimizer.apply_gradients(zip(gen_grads, model.generator.trainable_variables))

    # Backprop on Discriminator
    disc_grads = disc_tape.gradient(d_loss, model.discriminator.trainable_variables)
    model.disc_optimizer.apply_gradients(zip(disc_grads, model.discriminator.trainable_variables))

    return d_loss, adv_loss, content_loss


def train(model, dataset, log_iter, writer):
    """
    Function that defines a single training step for the AgingGAN.
    Args:
        model: An object that contains tf keras compiled generator and
               discriminator models.
        dataset: A tf data object that contains source and target domain images,
                 conditions and labels
        log_iter: Number of iterations after which to add logs in
                  tensorboard.
        writer: Summary writer
    """
    with writer.as_default():
        # Iterate over dataset
        for source_conditioned_img, true_img, true_condition, false_condition, true_label in dataset:
            disc_loss, adv_loss, content_loss = train_step(model,
                                                           source_conditioned_img,
                                                           true_img,
                                                           true_condition,
                                                           false_condition,
                                                           true_label)
            # Log tensorboard summaries if log iteration is reached.
            if model.iterations % log_iter == 0:
                tf.summary.scalar('Adversarial Loss', adv_loss, step=model.iterations)
                tf.summary.scalar('Content Loss', content_loss, step=model.iterations)
                tf.summary.scalar('Discriminator Loss', disc_loss, step=model.iterations)
                tf.summary.image('Input Image', tf.cast(255 * (source_conditioned_img[..., :3] + 1.0) / 2.0, tf.uint8),
                                 step=model.iterations)
                tf.summary.image('Generated',
                                 tf.cast(255 * (model.generator.predict(source_conditioned_img) + 1.0) / 2.0, tf.uint8),
                                 step=model.iterations)
                model.generator.save('models/generator.h5')
                model.discriminator.save('models/discriminator.h5')
                writer.flush()
            model.iterations += 1


@tf.function
def classifier_train_step(model, img, label):
    """
    Function that defines a single training step for the
    age classifier.
    Args:
        model: A model object that contains the age classifer.
        img: The input image tensors.
        label: The age category labels of the input images.
    """

    with tf.GradientTape() as tape:
        # Given image, predict label
        predicted_labels, _ = model.age_classifier(img)

        # Calculate the loss
        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=False)(label, predicted_labels)

    # Backprop the loss
    grads = tape.gradient(loss, model.age_classifier.trainable_variables)
    model.cls_optimizer.apply_gradients(zip(grads, model.age_classifier.trainable_variables))

    return loss, predicted_labels


def train_classifier(model, dataset, log_iter, writer):
    """
    Function that defines training for the Face classifier.
    Args:
        model: An object that contains tf keras compiled generator and
               discriminator models.
        dataset: A tf data object that contains image and age labels.
        log_iter: Number of iterations after which to add logs in
                  tensorboard.
        writer: Summary writer
    """
    acc = tf.metrics.SparseCategoricalAccuracy()
    with writer.as_default():
        for img, label in dataset:
            loss, pred = classifier_train_step(model, img, label)
            if model.iterations % log_iter == 0:
                acc.update_state(label, pred)
                tf.summary.scalar('Classifier Loss', loss, step=model.iterations)
                tf.summary.scalar('Accuracy', acc.result(), step=model.iterations)
                model.age_classifier.save('models/age_classifier.h5')
                acc.reset_states()
                writer.flush()
            model.iterations += 1


def main():
    # Parse the CLI arguments.
    args = parser.parse_args()

    # create directory for saving trained models.
    if not os.path.exists('models'):
        os.makedirs('models')

    if not os.path.exists('models/age_classifier.h5'):
        # Create the tensorflow dataset for age classification
        ds_classifier = DataLoaderAge(args.image_dir, args.text_dir, args.img_size).dataset(args.batch_size * 2)

        # Define the directory for saving the face classifier training tensorbaord summary.
        classifier_summary_writer = tf.summary.create_file_writer('logs/classifier')

        # Create the age classifier only
        gan = AgingGAN(args, True)

        # Run pre-training for the classifier.
        for _ in range(20):
            train_classifier(gan, ds_classifier, args.save_iter, classifier_summary_writer)

        # Clear session and start anew
        tf.keras.backend.clear_session()

    # Define the directory for saving the face aging gan training tensorbaord summary.
    train_summary_writer = tf.summary.create_file_writer('logs/gan')

    # Create the tensorflow dataset for cGAN
    ds_gan = DataLoaderGAN(args.image_dir, args.text_dir, args.img_size).dataset(args.batch_size)

    # Create the GAN model and load the pretrained age classifier
    gan = AgingGAN(args, False)

    # Train the GAN.
    for _ in range(args.epochs):
        train(gan, ds_gan, args.save_iter, train_summary_writer)


if __name__ == '__main__':
    main()
