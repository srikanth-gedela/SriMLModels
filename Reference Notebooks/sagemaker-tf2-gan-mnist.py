#!/usr/bin/env python
# coding: utf-8

# # SAGEMAKER TENSORFLOW2 GAN MNIST EXAMPLE


from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import keras
import argparse, os
import glob
import time





class GeneratorModel(tf.keras.Model):
    def __init__(self, input_shape=(100,)):
        super(GeneratorModel, self).__init__()
        
        self.l1_dense = tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=input_shape)
        self.l1_batchn = tf.keras.layers.BatchNormalization()
        self.l1_relu = tf.keras.layers.LeakyReLU()
        self.l1_reshape = tf.keras.layers.Reshape((7,7,256))
        
        self.l2_conv2dtr = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)
        self.l2_batchn = tf.keras.layers.BatchNormalization()
        self.l2_relu = tf.keras.layers.LeakyReLU()
        
        self.l3_conv2dtr = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.l3_batchn = tf.keras.layers.BatchNormalization()
        self.l3_relu = tf.keras.layers.LeakyReLU()
        
        self.l4_conv2dtr = tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    
    def build(self, input_shape=(100,)):
        super(GeneratorModel, self).build(input_shape)
                                              
    def call(self, x):
        x = self.l1_dense(x)
        x = self.l1_batchn(x)
        x = self.l1_relu(x)
        x = self.l1_reshape(x)

        x = self.l2_conv2dtr(x)
        x = self.l2_batchn(x)
        x = self.l2_relu(x)

        x = self.l3_conv2dtr(x)
        x = self.l3_batchn(x)
        x = self.l3_relu(x)
        
        return self.l4_conv2dtr(x)


class DiscriminatorModel(tf.keras.Model):
                                              
    def __init__(self, image_shape=(28,28,1)):
        super(DiscriminatorModel, self).__init__()
        self.l1_conv2d = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=image_shape)
        self.l1_relu = tf.keras.layers.LeakyReLU()
        self.l1_dropout = tf.keras.layers.Dropout(0.3)
        
        self.l2_conv2d = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')
        self.l2_relu = tf.keras.layers.LeakyReLU()
        self.l2_dropout = tf.keras.layers.Dropout(0.3)
        
        self.l3_flatten = tf.keras.layers.Flatten()
        self.l3_dense = tf.keras.layers.Dense(1)
        
        
    def build(self, input_shape=(28,28,1)):
        super(DiscriminatorModel, self).build(input_shape=input_shape)
    def call(self, x):
        x = self.l1_conv2d(x)
        x = self.l1_relu(x)
        x = self.l1_dropout(x)
        
        x = self.l2_conv2d(x)
        x = self.l2_relu(x)
        x = self.l2_dropout(x)
        
        x = self.l3_flatten(x)
        return self.l3_dense(x)

    
def model_runner(epochs=1, lr=1e-4, batch_size=128, gpu_count=1, model_dir="/tmp/model", training_dir="./data", image_shape=(28,28,1), noise_dim=100, num_examples_to_generate=16):
    

    buffer_size = batch_size*100

    
    path = os.path.join(training_dir, 'training.npz') 
    data = np.load(path, allow_pickle=True)
    #data: ((train_images, train_labels),(test_images,test_labels))
    train_images = data["data"][0][0]
    #(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], image_shape[0], image_shape[1], image_shape[2]).astype('float32')
    train_images = train_images[:buffer_size]
    train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size).batch(batch_size)

    generator = GeneratorModel(input_shape=(noise_dim,))
    discriminator = DiscriminatorModel(image_shape=image_shape)

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    seed = tf.random.normal([num_examples_to_generate, noise_dim])




    @tf.function
    def train_step(images):
        noise = tf.random.normal([batch_size, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        return gen_loss+disc_loss

    
    def train(dataset, epochs):
        losses_ = []
        for epoch in range(epochs):
            batch_loss = []
            for image_batch in dataset:
                loss_ = train_step(image_batch)
                batch_loss.append(loss_.numpy())
            losses_.append(np.mean(batch_loss))
            print ("epoch {0} loss {1}".format(epoch, losses_[-1]))
        return losses_

    losses_ = train(train_dataset, epochs)

    tf.saved_model.save(generator, os.path.join(model_dir, "001-generator"))
    tf.saved_model.save(discriminator, os.path.join(model_dir, "001-discriminator"))






#     checkpoint_dir = './gan_training_checkpoints'
#     checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
#     checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                      discriminator_optimizer=discriminator_optimizer,
#                                      generator=generator,
#                                      discriminator=discriminator)


#    parser = argparse.ArgumentParser()


if __name__ == '__main__':
    print ("tensorflow version:", tf.__version__)


    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    
    args, _ = parser.parse_known_args()
    epochs     = args.epochs
    lr         = args.learning_rate
    batch_size = args.batch_size
    gpu_count  = args.gpu_count
    model_dir  = args.model_dir
    training_dir   = args.training
    
    model_runner(epochs=epochs, lr=lr, batch_size=batch_size, gpu_count=gpu_count, model_dir=model_dir, training_dir=training_dir)
