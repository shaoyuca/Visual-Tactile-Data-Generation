import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras
import datetime
import os
import time
from matplotlib import pyplot as plt
from IPython import display
import numpy as np
import cv2
import librosa.display
from scipy import signal
import keras.backend as K
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

log_dir = "logs2/"
PATH = 'dataset/'


summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
# Train
f0 = [os.path.join(PATH, 'AcrylicGlass/visu', "%d.jpg" % i) for i in range(0, 1600)]
f1 = [os.path.join(PATH, 'compresswood/visu', "%d.jpg" % i) for i in range(0, 1600)]
f2 = [os.path.join(PATH, 'carbonfoil/visu', "%d.jpg" % i) for i in range(0, 1600)]
f3 = [os.path.join(PATH, 'carpet/visu', "%d.jpg" % i) for i in range(0, 1600)]
f4 = [os.path.join(PATH, 'finefoam/visu', "%d.jpg" % i) for i in range(0, 1600)]
f5 = [os.path.join(PATH, 'marble/visu', "%d.jpg" % i) for i in range(0, 1600)]
f6 = [os.path.join(PATH, 'finerubber/visu', "%d.jpg" % i) for i in range(0, 1600)]
f7 = [os.path.join(PATH, 'squared/visu', "%d.jpg" % i) for i in range(0, 1600)]
f8 = [os.path.join(PATH, 'leather/visu', "%d.jpg" % i) for i in range(0, 1600)]
filepaths = f0+f1+f2+f3+f4+f5+f6+f7+f8

f_0 = [os.path.join(PATH, 'AcrylicGlass/tact', "%d.npy" % i) for i in range(0, 1600)]
f_1 = [os.path.join(PATH, 'compresswood/tact', "%d.npy" % i) for i in range(0, 1600)]
f_2 = [os.path.join(PATH, 'carbonfoil/tact', "%d.npy" % i) for i in range(0, 1600)]
f_3 = [os.path.join(PATH, 'carpet/tact', "%d.npy" % i) for i in range(0, 1600)]
f_4 = [os.path.join(PATH, 'finefoam/tact', "%d.npy" % i) for i in range(0, 1600)]
f_5 = [os.path.join(PATH, 'marble/tact', "%d.npy" % i) for i in range(0, 1600)]
f_6 = [os.path.join(PATH, 'finerubber/tact', "%d.npy" % i) for i in range(0, 1600)]
f_7 = [os.path.join(PATH, 'squared/tact', "%d.npy" % i) for i in range(0, 1600)]
f_8 = [os.path.join(PATH, 'leather/tact', "%d.npy" % i) for i in range(0, 1600)]
filepaths_1 = f_0+f_1+f_2+f_3+f_4+f_5+f_6+f_7+f_8


# Test
m0 = [os.path.join(PATH, 'AcrylicGlass/visu', "%d.jpg" % i) for i in range(1800, 2000)]
m1 = [os.path.join(PATH, 'compresswood/visu', "%d.jpg" % i) for i in range(1800, 2000)]
m2 = [os.path.join(PATH, 'carbonfoil/visu', "%d.jpg" % i) for i in range(1800, 2000)]
m3 = [os.path.join(PATH, 'carpet/visu', "%d.jpg" % i) for i in range(1800, 2000)]
m4 = [os.path.join(PATH, 'finefoam/visu', "%d.jpg" % i) for i in range(1800, 2000)]
m5 = [os.path.join(PATH, 'marble/visu', "%d.jpg" % i) for i in range(1800, 2000)]
m6 = [os.path.join(PATH, 'finerubber/visu', "%d.jpg" % i) for i in range(1800, 2000)]
m7 = [os.path.join(PATH, 'squared/visu', "%d.jpg" % i) for i in range(1800, 2000)]
m8 = [os.path.join(PATH, 'leather/visu', "%d.jpg" % i) for i in range(1800, 2000)]
Afilepaths = m0+m1+m2+m3+m4+m5+m6+m7+m8

m_0 = [os.path.join(PATH, 'AcrylicGlass/tact', "%d.npy" % i) for i in range(1800, 2000)]
m_1 = [os.path.join(PATH, 'compresswood/tact', "%d.npy" % i) for i in range(1800, 2000)]
m_2 = [os.path.join(PATH, 'carbonfoil/tact', "%d.npy" % i) for i in range(1800, 2000)]
m_3 = [os.path.join(PATH, 'carpet/tact', "%d.npy" % i) for i in range(1800, 2000)]
m_4 = [os.path.join(PATH, 'finefoam/tact', "%d.npy" % i) for i in range(1800, 2000)]
m_5 = [os.path.join(PATH, 'marble/tact', "%d.npy" % i) for i in range(1800, 2000)]
m_6 = [os.path.join(PATH, 'finerubber/tact', "%d.npy" % i) for i in range(1800, 2000)]
m_7 = [os.path.join(PATH, 'squared/tact', "%d.npy" % i) for i in range(1800, 2000)]
m_8 = [os.path.join(PATH, 'leather/tact', "%d.npy" % i) for i in range(1800, 2000)]
Bfilepaths_1 = m_0+m_1+m_2+m_3+m_4+m_5+m_6+m_7+m_8


def my_func(txt):

    a = np.load(txt, allow_pickle=True)

    return a.astype(np.float32)


def load(txt_name, image_name):

    # load the processed spectrogram and image
    image = tf.io.read_file(image_name)
    image = tf.image.decode_jpeg(image, channels=1)
    txt = tf.compat.v1.py_func(my_func, [txt_name], tf.float32)
    txt = tf.reshape(txt, [256, 256, 1])

    input_image = tf.cast(image, tf.float32)
    real_data = tf.cast(txt, tf.float32)

    if tf.random.uniform(()) > 0.8:
        input_image = tf.image.flip_left_right(input_image)
    if tf.random.uniform(()) < 0.2:
        input_image = tf.image.flip_up_down(input_image)

    input_image = tf.image.random_contrast(input_image, lower=0, upper=1)
    input_image = tf.image.random_brightness(input_image, 0.1)
    
    input_image = (input_image / 127.5) - 1

    return real_data, input_image


BUFFER_SIZE = 410
BATCH_SIZE = 8

# training
image_name = tf.constant(filepaths)
txt_name = tf.constant(filepaths_1)
train_dataset = tf.data.Dataset.from_tensor_slices((txt_name, image_name))
train_dataset = train_dataset.map(load, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
showtrain_dataset = train_dataset.shuffle(500).batch(1)
train_dataset = train_dataset.batch(BATCH_SIZE)

# testing
image_name_1 = tf.constant(Afilepaths)
txt_name_1 = tf.constant(Bfilepaths_1)
test_dataset = tf.data.Dataset.from_tensor_slices((txt_name_1, image_name_1))
test_dataset = test_dataset.map(load, num_parallel_calls=tf.data.experimental.AUTOTUNE)
show_dataset = test_dataset.shuffle(800).batch(1)
test_dataset = test_dataset.batch(1)


sample_trainspec, sample_trainimage = next(iter(showtrain_dataset))
sample_spec, sample_image = next(iter(show_dataset))

#-----------------------------------------------Networks--------------------------------------------------------------#
# build generator
def _get_norm_layer(norm):
    if norm == 'none':
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return tf.keras.layers.BatchNormalization
    elif norm == 'instance_norm':
        return tfa.layers.InstanceNormalization
    elif norm == 'layer_norm':
        return tf.keras.layers.LayerNormalization

def ResnetGenerator(input_shape=(256, 256, 1),
                    label_shape=(8, 8, 1024),
                    output_channels=1,
                    dim=32,
                    n_downsamplings=4,
                    n_blocks=9,
                    norm='instance_norm'):
    Norm = _get_norm_layer(norm)
    initializer = tf.keras.initializers.glorot_normal

    # residual layers
    def _residual_block(x):
        dim = x.shape[-1]
        h = x

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = tf.keras.layers.Conv2D(dim, 3, padding='valid', kernel_initializer=initializer, use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = tf.keras.layers.Conv2D(dim, 3, padding='valid', kernel_initializer=initializer, use_bias=False)(h)
        h = Norm()(h)

        return tf.keras.layers.add([x, h])
    # Inputs
    h = inputs = tf.keras.Input(shape=input_shape)
    label = tf.keras.Input(shape=label_shape)
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = tf.keras.layers.Conv2D(dim, 7, padding='valid', kernel_initializer=initializer, use_bias=False)(h)
    h = tf.nn.relu(h)

    # downsampling
    dim_ = dim
    for _ in range(n_downsamplings):
        dim = min(dim * 2, dim_ * 16)
        h = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)
        if _ == 0:
            x1 = h
        if _ == 1:
            x2 = h
        if _ == 2:
            x3 = h

    # label information
    x = tf.keras.layers.Conv2DTranspose(512, 4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(label)
    x = Norm()(x)
    x = tf.nn.relu(x)
    h = tf.keras.layers.concatenate([h, x], axis=3, name='concatenation')  # 16, 16, 1024

    # residue fusion
    for _ in range(n_blocks):
        h = _residual_block(h)

    # upsampling
    for _ in range(n_downsamplings):
        dim //= 2
        h = tf.keras.layers.Conv2DTranspose(dim, 4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

        if _ == 0:
            h = tf.keras.layers.concatenate([x3, h], axis=3)
        if _ == 1:
            h = tf.keras.layers.concatenate([x2, h], axis=3)
        if _ == 2:
            h = tf.keras.layers.concatenate([x1, h], axis=3)


    # last layer
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = tf.keras.layers.Conv2D(output_channels, 7, kernel_initializer=initializer, padding='valid')(h)
    h = tf.tanh(h)

    model = tf.keras.Model(inputs=[inputs, label], outputs=h)
    model.summary()

    return model

generator = ResnetGenerator()

# Generator loss
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# Discriminator
def Discriminator(dim=64,
                  n_downsamplings=3,
                  norm='layer_norm'):
    dim_ = dim
    Norm = _get_norm_layer(norm)
    initializer = tf.keras.initializers.glorot_normal
    inp = tf.keras.layers.Input((256, 256, 1))
    tar = tf.keras.layers.Input((256, 256, 1))


    h = tf.keras.layers.concatenate([inp, tar], axis=3)
    h = tf.keras.layers.Conv2D(dim, 4, strides=2, kernel_initializer=initializer, padding='same')(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)
    layers = []
    layers.append(h)

    for _ in range(n_downsamplings - 1):
        dim = min(dim * 2, dim_ * 16)
        h = tf.keras.layers.Conv2D(dim, 4, strides=2, kernel_initializer=initializer, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.leaky_relu(h, alpha=0.2)
        layers.append(h)


    dim = min(dim * 2, dim_ * 16)
    h = tf.keras.layers.Conv2D(dim, 4, strides=1, kernel_initializer=initializer, padding='same', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)
    layers.append(h)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(h)
    h = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1)
    h = Norm()(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)
    layers.append(h)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(h)
    h = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)
    layers.append(h)

    model = tf.keras.Model(inputs=[inp, tar], outputs=[layers[0], layers[1], layers[1], layers[3], layers[4], layers[5]])
    model.summary()

    return model


discriminator = Discriminator()

def discriminator_loss(disc_real_output, disc_generated_output):

    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = 0.5 * (real_loss + generated_loss)

    return total_disc_loss

# FM loss
def fm_loss(gen_activations, real_activations):
    total_loss = 0
    for g, r in zip(gen_activations, real_activations):
        total_loss += tf.reduce_mean(tf.abs(g - r))

    return total_loss

# VGG loss
def vgg_loss(fake, real):

    total_loss = 0
    for g, r in zip(fake, real):
        total_loss += tf.reduce_mean(tf.square(g - r))

    return total_loss

# label information
def Class_feature():

    # base model
    shape = (256, 256, 3)
    base_model = tf.keras.applications.DenseNet121(input_shape=shape, include_top=False, weights=None)
    base_model.trainable = False
    flatten = tf.keras.layers.GlobalAveragePooling2D()
    dropout_1 = tf.keras.layers.Dropout(0.9)
    dense = tf.keras.layers.Dense(128, activation='relu')
    prediction_layer = tf.keras.layers.Dense(9)

    model = tf.keras.Sequential([
        base_model,
        flatten,
        dropout_1,
        dense,
        prediction_layer])

    model.trainable = False
    model.summary()

    model.load_weights('./pretrain/densetactile.h5')

    # feature layer
    feature_layer_model = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)
    feature_layer_model.trainable = False
    feature_layer_model.summary()

    return feature_layer_model

desnet = Class_feature()


# perceptual layer
def vgg54():
    shape = (256, 256, 3)
    base_model = tf.keras.applications.VGG19(input_shape=shape, include_top=False, weights='imagenet')
    base_model.trainable = False

    selectedLayers = [4, 5, 17, 18]  # (1,2,9,10,17,18)
    selectedOutputs = [base_model.layers[i].output for i in selectedLayers]
    feature_layer_model = keras.Model(inputs=base_model.input, outputs=selectedOutputs)
    feature_layer_model.trainable = False
    feature_layer_model.summary()

    return feature_layer_model

vgg = vgg54()

# Optimizer
generator_optimizer = tf.keras.optimizers.Adam(0.0002)
discriminator_optimizer = tf.keras.optimizers.Adam(0.0002)
checkpoint_dir = './model'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


def generate_feature(model, input):

    input = input * 0.5 + 0.5
    new_input = tf.concat([input, input, input], axis=3)
    class_feature = model(new_input, training=True)

    return class_feature


def generate_output(model, test_input):

    class_feature = generate_feature(desnet, test_input)
    prediction = model([test_input, class_feature], training=True)

    return prediction


# data prediction and visualization
def generate_images(model, test_input, tar, i):

    class_feature = generate_feature(desnet, test_input)
    start = datetime.datetime.now()
    prediction = model([test_input, class_feature], training=True)
    end = datetime.datetime.now()
    print('Time taken for generating is {} sec\n'.format(end - start))
    generated = np.reshape(prediction[0], [256, 256])
    generated_show = generated * 0.5 + 0.5

    # visualize
    plt.figure(figsize=(3, 3), dpi=100)
    plt.title('input spectrogram')
    inp_spec = tf.reshape(test_input, [256, 256])
    input = inp_spec.numpy()
    input = np.exp(input)
    librosa.display.specshow(librosa.amplitude_to_db(input, ref=np.max))
    plt.tight_layout()
    plt.show()

    plt.subplot(121)
    plt.title('Generated results')
    plt.imshow(generated_show, cmap='gray')
    plt.axis('off')

    plt.subplot(122)
    plt.title('Real image')
    image = tf.reshape(tar, [256, 256])
    image = image.numpy()
    plt.imshow(image * 0.5 + 0.5, cmap='gray')
    plt.axis('off')
    plt.show()


# data save
def data_save(model, test_input, i):
    class_feature = generate_feature(desnet, test_input)
    start = datetime.datetime.now()
    prediction = model([test_input, class_feature], training=True)
    end = datetime.datetime.now()
    print('Time taken for generating is {} sec\n'.format(end - start))

    inp_spec = np.reshape(prediction[0], [256, 256])  # -1 - 1
    inp_spec = inp_spec * 0.5 + 0.5
    inp_spec = inp_spec * 255.0
    cv2.imwrite("generated_images/{}.npy".format(i), inp_spec)


gpl = 10
@tf.function
def w_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)

# cgan
def gradient_penalty(input, real, fake):
    alpha = tf.random.uniform(shape=[len(real), 1, 1, 1], minval=0., maxval=1.)
    interpolated = alpha * real + (1 - alpha) * fake

    with tf.GradientTape() as tape_p:
        tape_p.watch(interpolated)
        log = discriminator([input, interpolated], training=True)
        logit = log[5]

    grad = tape_p.gradient(logit, interpolated)
    grad_norm = tf.norm(tf.reshape(grad, (BATCH_SIZE, -1)), axis=1)

    return gpl * tf.reduce_mean(tf.square(grad_norm - 1.))


def train_step_gen(input_spec, target, train_show, test_show, epoch):

    new = input_spec * 0.5 + 0.5
    new_input = tf.concat([new, new, new], axis=-1)
    class_vector = desnet(new_input, training=False)

    with tf.GradientTape() as gen_tape:

        gen_output = generator([input_spec, class_vector], training=True)

        # cgan & gan
        # Two outputs dis + FM
        disc_generated = discriminator([input_spec, gen_output], training=True)
        disc_generated_output = disc_generated[5]
        disc_real = discriminator([input_spec, target], training=True)

        # w_loss
        gen_gan_loss = w_loss(disc_generated_output, -tf.ones(shape=tf.shape(disc_generated_output), dtype=tf.float32))

        gen_new_output = gen_output * 0.5 + 0.5
        target_new = target * 0.5 + 0.5
        gen_new = tf.concat([gen_new_output, gen_new_output, gen_new_output], axis=3)
        tar_new = tf.concat([target_new, target_new, target_new], axis=3)

        # vgg loss
        gen_vgg_out = vgg(gen_new, training=False)
        real_vgg_out = vgg(tar_new, training=False)
        gen_vgg_loss = vgg_loss(gen_vgg_out, real_vgg_out)

        # FM loss
        gen_fm_loss = fm_loss(disc_generated, disc_real)

        # total loss
        gen_total_loss = gen_gan_loss + 1.0 * gen_vgg_loss + 10.0 * gen_fm_loss

    generator_gradients = gen_tape.gradient(target=gen_total_loss, sources=generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))


    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('w_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('FM_loss', gen_fm_loss, step=epoch)
        tf.summary.scalar('gen_vgg_loss', gen_vgg_loss, step=epoch)
        tf.summary.image('real training spec', tf.math.exp(sample_trainspec) * 0.5, step=epoch, max_outputs=3)
        tf.summary.image('fake training spec', tf.math.exp(train_show) * 0.5, step=epoch, max_outputs=3)
        tf.summary.image('real training image', sample_trainimage * 0.5 + 0.5, step=epoch, max_outputs=3)
        tf.summary.image('real testing spec', tf.math.exp(sample_spec) * 0.5, step=epoch, max_outputs=6)
        tf.summary.image('fake testing spec', tf.math.exp(test_show) * 0.5, step=epoch, max_outputs=6)
        tf.summary.image('real testing image', sample_image * 0.5 + 0.5, step=epoch, max_outputs=3)

    return gen_total_loss, gen_gan_loss, gen_fm_loss, gen_vgg_loss


# discriminator train
def train_step_dis(input_spec, target, epoch):

    new = input_spec * 0.5 + 0.5
    new_input = tf.concat([new, new, new], axis=-1)
    class_vector = desnet(new_input, training=False)

    with tf.GradientTape() as disc_tape:
        gen_output = generator([input_spec, class_vector], training=True)

        # cgan & gan
        disc_real = discriminator([input_spec, target], training=True)
        disc_real_output = disc_real[5]
        disc_generated = discriminator([input_spec, gen_output], training=True)
        disc_generated_output = disc_generated[5]
        y_true = tf.ones(shape=tf.shape(disc_real_output), dtype=tf.float32)

        real_loss = w_loss(-y_true, disc_real_output)
        fake_loss = w_loss(y_true, disc_generated_output)

        # cgan & gan
        gp_loss = gradient_penalty(input_spec, target, gen_output)
        disc_loss = 0.5 * (real_loss + fake_loss) + gp_loss

    discriminator_gradients = disc_tape.gradient(target=disc_loss, sources=discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gp_loss', gp_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)

    return disc_loss, gp_loss


# training process
def fit(train_ds, epochs, test_ds):
  for epoch in range(epochs):
    start = time.time()
    display.clear_output(wait=True)
    print("Epoch: ", epoch)

    for n, (input_spec, target) in train_ds.enumerate():
      print('.', end='')

      sample_trainspec, sample_trainimage = next(iter(train_dataset))
      sample_spec, sample_image = next(iter(show_dataset))

      gt_image = generate_output(generator, sample_trainimage)
      g_image = generate_output(generator, sample_image)

      D, GP = train_step_dis(input_spec, target, epoch)
      TL, G, FM, VGG = train_step_gen(input_spec, target, gt_image, g_image, epoch)

      print("D_loss: {:.2f}".format(D), "GP_loss: {:.2f}".format(GP), "G_loss: {:.2f}".format(G), "gen_fm_loss: {:.2f}".format(FM),
            "gen_vgg_loss: {:.2f}".format(VGG), "gen_total_loss: {:.2f}".format(TL))
    print()

    # Saving (checkpoint) the model every 20 epochs
    if (epoch + 1) % 20 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))
    print()
  checkpoint.save(file_prefix = checkpoint_prefix)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, dest='epoch', default=400)
    parser.add_argument('--train', help='train', action='store_true')
    parser.add_argument('--test', help='test', action='store_true')
    parser.add_argument('--visualize', help='visualize result', action='store_true')
    args = parser.parse_args()

    if (args.train):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        fit(train_dataset, args.epoch, test_dataset)

    if (args.test):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        i = 0
        for inp, tar in test_dataset.take(1800):
           data_save(generator, inp, i)
           i=i+1

    if (args.visualize):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        i = 0
        for inp, tar in show_dataset.take(1800):
            generate_images(generator, inp, tar, i)
            i=i+1


if __name__=='__main__':
    main()
