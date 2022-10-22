import tensorflow as tf
import PIL


def compute_l1_loss(y_true, y_false):
  return tf.reduce_mean(tf.math.abs(y_true - y_false))

def no_gan_inner_step(gt_images,
                      lq_images,
                      main_model,
                      optimizer):
  with tf.GradientTape() as tape:
    generated_images = main_model(lq_images)

    loss = compute_l1_loss(gt_images, generated_images)

  gradients = tape.gradient(loss, main_model.trainable_variables)

  optimizer.apply_gradients(zip(gradients, main_model.trainable_variables))

  return loss