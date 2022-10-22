from typing_extensions import Self
import tensorflow as tf
import tensorflow_addons as tfa


class ESRGAN(tf.keras.Model):
    def __init__(self):
        super(ESRGAN, self).__init__()

        self.input_conv = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding="SAME")
        self.rrdb1 = RRDB()
        self.rrdb2 = RRDB()
        self.rrdb3 = RRDB()
        self.rrdb4 = RRDB()
        self.rrdb5 = RRDB()
        self.rrdb6 = RRDB()
        self.second_conv = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding="SAME")

        self.upsample1 = UpSamplingBlock()

        self.reconstruction_conv1 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding="SAME")
        self.final_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.final_reconstruction_conv = tf.keras.layers.Conv2D(1, (3, 3), strides=(1, 1), padding="SAME")

    def call(self, x):
        # initial conv
        x1 = self.input_conv(x)
        # Residual in Residual Dense Blocks layers
        x2 = self.rrdb1(x1)
        x3 = self.rrdb2(x2)
        x4 = self.rrdb3(x3)
        x5 = self.rrdb4(x4)
        x6 = self.rrdb5(x5)
        x7 = self.rrdb6(x6)

        # upsampling section
        x8 = x1 + x7
        x9 = self.upsample1(x8)

        # final reconstruction
        x10 = self.reconstruction_conv1(x9)
        x11 = self.final_lrelu(x10)
        x12 = self.final_reconstruction_conv(x11)

        return x12


class RRDB(tf.keras.Model):
    def __init__(self):
        super(RRDB, self).__init__()
        self.rdb1 = RDB()
        self.rdb2 = RDB()
        self.rdb3 = RDB()

    def call(self, x):
        x1 = self.rdb1(x)
        x2 = self.rdb2(x1)
        x3 = self.rdb3(x2)

        # return input scaled with the output of the block
        return x3 * 0.2 + x



class RDB(tf.keras.Model):
    def __init__(self):
        super(RDB, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding="SAME")
        self.conv2 = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding="SAME")
        self.conv3 = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding="SAME")
        self.conv4 = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding="SAME")
        self.final_conv = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding="SAME")
        
        self.activ1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.activ2 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.activ3 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.activ4 = tf.keras.layers.LeakyReLU(alpha=0.2)

    def call(self, x):
        x1 = self.activ1(self.conv1(x))
        x2 = self.activ2(self.conv2(tf.keras.layers.concatenate([x, x1])))
        x3 = self.activ3(self.conv3(tf.keras.layers.concatenate([x, x1, x2])))
        x4 = self.activ4(self.conv4(tf.keras.layers.concatenate([x, x1, x2, x3])))
        x5 = self.final_conv(tf.keras.layers.concatenate([x, x1, x2, x3, x4]))

        # return input scaled with the output of the block
        return x5 * 0.2 + x


class UpSamplingBlock(tf.keras.Model):
    def __init__(self):
        super(UpSamplingBlock, self).__init__()

        #Conv2D layers
        self.upsampl1 = tf.keras.layers.UpSampling2D()
        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding="SAME")
        self.activ1 = tf.keras.layers.LeakyReLU(alpha=0.2)

    def call(self, x):
        x1 = self.upsampl1(x)
        x2 = self.conv1(x1)
        x3 = self.activ1(x2)

        # return input scaled with the output of the block
        return x3

