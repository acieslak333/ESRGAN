from model import ESRGAN
from train import *
from data_loader import *
import tensorflow as tf
import glob
import os 
import time



@tf.function
def train_step(gt_images, lq_images, model, optimizer):
  return no_gan_inner_step(gt_images,
                      lq_images,
                      model,
                      optimizer)


def train(epochs, train_generator, model, train_steps):
    print("Start Training")

    train_loss_metric = tf.keras.metrics.Mean()
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.9, beta_2=0.99)
    loss_results = []
  
    checkpoint_dir = './ckpts'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

    for epoch in range(0, epochs):
        train_loss_metric.reset_states()
        epoch_time = time.time()
        batch_time = time.time()
        step = 0

        epoch_count = f"0{epoch + 1}/{epochs}" if epoch < 9 else f"{epoch + 1}/{epochs}"

        # for img, first_kernel, second_kernel, sinc_kernel in train_generator:
        #     gt_img, lq_img = feed_data(img, first_kernel, second_kernel, sinc_kernel, [feed_props_1, feed_props_2])
        #     gt_img, lq_img = pool_train_data.get_pool_data(gt_img, lq_img)
        #     gt_img = usm_sharpener.sharp(gt_img)
        #     loss = train_step(gt_img, lq_img, model, optimizer)

        #     print('\r', 'Epoch', epoch_count, '| Step', f"{step}/{train_steps}",
        #         '| Loss:', f"{loss:.5f}", "| Step Time:", f"{time.time() - batch_time:.2f}", end='')  
            
        #     train_loss_metric.update_state(loss)
        #     loss = train_loss_metric.result().numpy()
        #     step += 1

        #     loss_results.append(loss)

        #     batch_time = time.time()

        for img in train_generator:
            # print(type())
            new_shape = list(img.shape)
            new_shape = (new_shape[1]*2 , new_shape[2]*2)

            bimg = tf.image.resize(img, new_shape)
            loss = train_step(bimg, img, model, optimizer)

            print('\r', 'Epoch', epoch_count, '| Step', f"{step}/{train_steps}",
                '| Loss:', f"{loss:.5f}", "| Step Time:", f"{time.time() - batch_time:.2f}", end='')  
            
            train_loss_metric.update_state(loss)
            loss = train_loss_metric.result().numpy()
            step += 1

            loss_results.append(loss)

            batch_time = time.time()

        checkpoint.save(file_prefix=checkpoint_prefix)
        # ema_api.compute_ema_weights(no_gan_model)
        # ema_checkpoint.save(file_prefix=ema_checkpoint_prefix)

        print('\r', 'Epoch', epoch_count, '| Step', f"{step}/{train_steps}",
                '| Loss:', f"{loss:.5f}", "| Epoch Time:", f"{time.time() - epoch_time:.2f}")
                

def main():
    
    epochs = 1
    batch_size = 1
    

    data_path = os.path.abspath("./data/train_images/*.png")
    train_images_paths = sorted(glob.glob(data_path))


    train_steps = int(len(train_images_paths) // batch_size)
    train_generator = get_train_dataset(batch_size, train_images_paths)

    model = ESRGAN()
    model.build((None, 256, 256, 3))
    print(model.summary())

    train(epochs=epochs, train_generator=train_generator,
     model=model, train_steps=train_steps)

if __name__ == "__main__":
    main()