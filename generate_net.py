import os
import sys
import numpy as np
from tensorflow.keras.preprocessing import text, sequence, image
from tensorflow.keras.datasets import imdb, mnist, cifar10
from tensorflow.keras import models, layers, optimizers, losses, activations, Input, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.applications import InceptionV3, VGG19
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import binary_crossentropy
import matplotlib.pyplot as plt


def sample(pred, temperature=1.0):
    pred = np.array(pred, dtype=np.float)
    pred = np.log(pred) / temperature
    exp_pred = np.exp(pred)
    pred = exp_pred / np.sum(exp_pred)
    prob = np.random.multinomial(1, pred, 1)
    return np.argmax(prob)


def text_generate_demo():
    with open('data/nietzsche.txt','r') as f:
        nicai_text = f.read()

    max_len = 60
    step = 3
    sentences = []
    next_char = []

    for i in range(0, len(nicai_text) - max_len, step):
        sentences.append(nicai_text[i:i+max_len])
        next_char.append(nicai_text[i+max_len])

    all_chars = sorted(list(set(nicai_text)))
    char_dict = {c:all_chars.index(c) for c in all_chars}

    x_train = np.zeros((len(sentences), max_len, len(all_chars)), dtype=np.bool)
    y_train = np.zeros((len(sentences), len(all_chars)), dtype=np.bool)
    for i, s in enumerate(sentences):
        for k, c in enumerate(s):
            x_train[i, k, char_dict[c]] = 1
        y_train[i, char_dict[next_char[i]]] = 1

    in_layer = Input(shape=(max_len, len(all_chars)))
    out = layers.LSTM(128)(in_layer)
    out = layers.Dense(len(all_chars), activation=activations.softmax)(out)
    model = Model(in_layer, out)
    model.summary()
    model.compile(optimizer=optimizers.RMSprop(lr=0.01), loss=losses.CategoricalCrossentropy())

    # for ep in range(60):
    #     print()
    #     print('epoch: ', ep)
    #     print()
    #     model.fit(x_train, y_train, batch_size=128, epochs=1)
    #     start_idx = np.random.randint(0, len(nicai_text) - max_len - 1)
    #     ready_text = nicai_text[start_idx:start_idx+max_len]
    #     print()
    #     print('ready text:')
    #     print(ready_text)
    #     for t in [0.2, 0.5, 1.0, 1.2]:
    #         print()
    #         print('--------temperature: ', t)
    #         print()
    #         sys.stdout.write(ready_text)
    #         for i in range(400):
    #             sampled = np.zeros((1, max_len, len(all_chars)))
    #             for idx, c in enumerate(ready_text):
    #                 sampled[0, idx, char_dict[c]] = 1
    #             pred = model.predict(sampled, verbose=0)[0]
    #             next_idx = sample(pred, t)
    #             next_char = all_chars[next_idx]
    #             ready_text += next_char
    #             ready_text = ready_text[1:]
    #             sys.stdout.write(next_char)

    model.fit(x_train, y_train, batch_size=128, epochs=60)

    start_idx = np.random.randint(0, len(nicai_text) - max_len - 1)
    ready_text = nicai_text[start_idx:start_idx+max_len]
    print()
    print('ready text:')
    print(ready_text)
    for t in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('--------temperature: ', t)
        print()
        sys.stdout.write(ready_text)
        for i in range(400):
            sampled = np.zeros((1, max_len, len(all_chars)))
            for idx, c in enumerate(ready_text):
                sampled[0, idx, char_dict[c]] = 1
            pred = model.predict(sampled, verbose=0)[0]
            next_idx = sample(pred, t)
            next_char = all_chars[next_idx]
            ready_text += next_char
            ready_text = ready_text[1:]
            sys.stdout.write(next_char)



def deep_dream_demo():
    import tensorflow
    tensorflow.compat.v1.disable_eager_execution()

    def resize_img(img, size):
        import scipy.ndimage
        img = np.copy(img)
        factors = (1, float(size[0]) / img.shape[1], float(size[1]) / img.shape[2], 1)
        return scipy.ndimage.zoom(img, factors, order=1)

    def deprocess_img(x):
        if K.image_data_format() == 'channels_first':
            x = x.reshape((3, x.shape[2], x.shape[3]))
            x = x.transpose((1, 2, 0))
        else:
            x = x.reshape((x.shape[1], x.shape[2], 3))
        x /= 2.
        x += 0.5
        x *= 255.
        x = np.clip(x, 0, 255).astype(np.uint8)
        return x

    def save_img(img, fname):
        from PIL import Image
        pil_img = deprocess_img(np.copy(img))
        Image.fromarray(pil_img).save(fname)

    def preprocess_img(img_path):
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        img = image.load_img(img_path)
        img = image.img_to_array(img)
        img = img.reshape((1,) + img.shape)
        img = preprocess_input(img)
        return img

    K.set_learning_phase(0)
    model = InceptionV3(weights='imagenet', include_top=False)
    model.summary()

    layer_contribute = {
        'mixed2': 0.2,
        'mixed3': 3.,
        'mixed4': 2.,
        'mixed5': 1.5,
        'mixed9_0': 3.,
        'mixed10': 2.
    }
    layer_dict = {l.name:l for l in model.layers}

    loss = K.variable(0.)
    for layer_name in layer_contribute.keys():
        power = layer_contribute[layer_name]
        act = model.get_layer(layer_name).output
        scaling = K.prod(K.cast(K.shape(act), np.float))
        loss = loss + (power * K.sum(K.square(act[:, 2:-2, 2:-2, :])) / scaling)
        # loss = loss.assign_add(power * K.sum(K.square(act[:, 2:-2, 2:-2, :])) / scaling)

    dream = model.input
    grads = K.gradients(loss, dream)[0]
    grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)
    fetch_loss_grads = K.function([dream], [loss, grads])

    def eval_loss_and_grads(x):
        a = fetch_loss_grads(x)
        return a[0], a[1]

    def grad_ascent(x, iter_num, step, max_loss=None):
        for i in range(iter_num):
            l, g = eval_loss_and_grads(x)
            if max_loss is not None and l > max_loss:
                break
            x += step * g
        return x

    step = 0.01
    num_octave = 3
    octave_scale = 1.4
    iter_num = 20
    max_loss = 10.
    src_img_path = 'image/elephant.png'

    img = preprocess_img(src_img_path)
    origin_shape = img.shape[1:3]
    successive_shapes = [origin_shape]
    for i in range(1, num_octave):
        shape = tuple([int(dim / (octave_scale ** i)) for dim in origin_shape])
        successive_shapes.append(shape)
    successive_shapes = successive_shapes[::-1]

    origin_img = np.copy(img)
    shrunk_origin_img = resize_img(img, successive_shapes[0])
    for shape in successive_shapes:
        img = resize_img(img, shape)
        img = grad_ascent(img, iter_num=iter_num, step=step, max_loss=max_loss)
        upscale_shrunk_origin_img = resize_img(shrunk_origin_img, shape)
        same_size_origin_img = resize_img(origin_img, shape)
        lost_detail = same_size_origin_img - upscale_shrunk_origin_img
        img += lost_detail
        shrunk_origin_img = resize_img(origin_img, shape)
        # save_img(img, 'image/hyt_dream_scale_' + str(shape) + '.jpg')
    save_img(img, 'image/elephant_final_dream.jpg')


def style_transform_demo(target_img_path, style_img_path, out_img_path, target_img_height=400, iter_num=30, save_per_iter=None):
    import tensorflow
    tensorflow.compat.v1.disable_eager_execution()

    from tensorflow.keras.applications.vgg19 import preprocess_input
    from scipy.optimize import fmin_l_bfgs_b
    from PIL import Image

    # target_img_path = 'image/hyt.jpg'
    # style_img_path = 'image/style4.png'
    # 加载目标图像，获取宽高
    width, height = image.load_img(target_img_path).size
    target_filename = os.path.basename(target_img_path)

    # 计算网络最终生成图的宽高
    img_height = target_img_height
    img_width = int(width * img_height / height)

    # 预处理图片
    def preprocess_img(im_path):
        img = image.load_img(im_path, target_size=(img_height, img_width))
        img = image.img_to_array(img)
        # 增加一个维度，作为batch
        img = img.reshape((1,) + img.shape)
        # 下面这个函数，用于处理图像，使其适用于keras的VGG19
        img = preprocess_input(img)
        return img

    # 将经过fmin_l_bfgs_b处理后的图片，进行还原
    def deprocess_img(x):
        # 三个通道都加指定值
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        # 调整通道顺序
        x = x[:, :, ::-1]
        # 限制数值范围
        x = np.clip(x, 0, 255).astype(np.uint8)
        return x

    # 开始定义网络
    # 定义目标图片和风格图片的常量，用于进行损失计算
    target_img = K.constant(preprocess_img(target_img_path))
    style_img = K.constant(preprocess_img(style_img_path))
    # 定义输入图片（生成图片）的占位符
    res_img = K.placeholder((1, img_height, img_width, 3))
    # 将三个图片依据batch进行拼接
    input_tensor = K.concatenate([target_img, style_img, res_img], axis=0)
    # 创建一个VGG19的无头模型，输入拼接后的batch
    model = VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
    model.summary()

    # 计算内容损失值的函数，即目标图片和生成图片的每个像素值差的平方再求和
    def content_loss(base, comba):
        return K.sum(K.square(comba - base))

    # 求不同通道见的相似度矩阵（格莱姆矩阵）
    def gram_matrix(x):
        # 将宽高放在最后两个维度，然后依据批展平（通道维度展平）
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
        # 相当于对不同通道的每个对应像素，求内积，即的到不同通道之间的相似性
        gram = K.dot(features, K.transpose(features))
        return gram

    # 计算风格损失
    def style_loss(base, comba):
        # 求风格图不同通道间的相似度矩阵
        B = gram_matrix(base)
        # 求生成图不同通道间的相似度矩阵
        C = gram_matrix(comba)
        channels = 3
        size = img_width * img_height
        # 根据公式计算
        return K.sum(K.square(B - C)) / (4. * (channels ** 2) * (size ** 2))

    # 计算像素间差别损失
    def total_var_loss(x):
        # 左上和左下像素值差的平方
        a = K.square(
            x[:, :img_height-1, :img_width-1, :] - x[:, 1:, :img_width-1, :]
        )
        # 左上和右上像素值差的平方
        b = K.square(
            x[:, :img_height-1, :img_width-1, :] - x[:, :img_height-1, 1:, :]
        )
        # 取个指数再求和
        return K.sum(K.pow(a + b, 1.25))

    # 构建VGG19网络中所有层名称和输出的对应字典
    layer_output_dict = {layer.name: layer.output for layer in model.layers}
    # 所选择内容层的名称
    content_layer_name = 'block5_conv2'
    # 所选择风格层的名称列表
    style_layer_names = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    # 配置损失权重
    content_weight = 0.025
    style_weight = 1.
    total_var_weight = 1e-4
    # 定义损失
    loss = K.variable(0.)
    # 取得内容层特征图的输出
    content_layer_out = layer_output_dict[content_layer_name]
    content_target_out = content_layer_out[0, :, :, :]
    content_res_out = content_layer_out[2, :, :, :]
    # 计算内容损失
    loss = loss + (content_weight * content_loss(content_target_out, content_res_out))
    # 累计计算风格损失
    for lname in style_layer_names:
        style_layer_out = layer_output_dict[lname]
        style_style_out = style_layer_out[1, :, :, :]
        style_res_out = style_layer_out[2, :, :, :]
        l = style_loss(style_style_out, style_res_out)
        loss = loss + (style_weight / len(style_layer_names) * l)
    # 计算像素间损失
    loss = loss + total_var_weight * total_var_loss(res_img)
    # 计算生成图对损失的梯度
    grads = K.gradients(loss, res_img)[0]
    # 定义传播函数
    fetch_loss_and_grads = K.function([res_img], [loss, grads])

    class Evaluator:

        def __init__(self):
            self.loss_value = None
            self.grads_value = None

        def loss(self, x):
            assert self.loss_value is None
            x = x.reshape((1, img_height, img_width, 3))
            out = fetch_loss_and_grads([x])
            loss_value = out[0]
            grads_value = out[1].flatten().astype(np.float)
            self.loss_value = loss_value
            self.grads_value = grads_value
            return self.loss_value

        def grads(self, x):
            assert self.grads_value is not None
            grads_value = np.copy(self.grads_value)
            self.loss_value = None
            self.grads_value = None
            return grads_value

    evaluator = Evaluator()

    if save_per_iter is not None:
        if not os.path.isdir(out_img_path):
            os.makedirs(out_img_path)

    x = preprocess_img(target_img_path)
    x = x.flatten()
    for i in range(iter_num):
        print('iter: ', i)
        # 使用fmin_l_bfgs_b算法进行计算
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x, fprime=evaluator.grads, maxfun=20)
        print('loss: ', min_val)
        if save_per_iter is not None and (i + 1) % save_per_iter == 0:
            saved_path = os.path.join(out_img_path, target_filename + '_iter' + str((i + 1)) + '.jpg')
            img = x.copy().reshape((img_height, img_width, 3))
            img = deprocess_img(img)
            Image.fromarray(img).save(saved_path)
    # 调整维度、输出保存
    if save_per_iter is None:
        img = x.copy().reshape((img_height, img_width, 3))
        img = deprocess_img(img)
        Image.fromarray(img).save(out_img_path)


def vae_demo():
    import tensorflow
    from scipy.stats import norm
    tensorflow.compat.v1.disable_eager_execution()

    img_shape = (28, 28, 1)
    batch_size = 16
    latent_dim = 2

    input_layer = Input(shape=img_shape)
    out = layers.Conv2D(32, 3, padding='same', activation=activations.relu)(input_layer)
    out = layers.Conv2D(64, 3, padding='same', activation=activations.relu, strides=2)(out)
    out = layers.Conv2D(64, 3, padding='same', activation=activations.relu)(out)
    out = layers.Conv2D(64, 3, padding='same', activation=activations.relu)(out)
    shape_before_flatten = K.int_shape(out)
    out = layers.Flatten()(out)
    out = layers.Dense(32, activation=activations.relu)(out)
    z_mean = layers.Dense(latent_dim)(out)
    z_log_var = layers.Dense(latent_dim)(out)
    def sampling(args):
        zm, zv = args
        epsilon = K.random_normal(shape=(K.shape(zm)[0], latent_dim), mean=0., stddev=1.)
        return zm + K.exp(zv) * epsilon
    z = layers.Lambda(sampling)([z_mean, z_log_var])

    decode_input_layer = Input(shape=K.int_shape(z)[1:])
    out = layers.Dense(np.prod(shape_before_flatten[1:]), activation=activations.relu)(decode_input_layer)
    out = layers.Reshape(shape_before_flatten[1:])(out)
    out = layers.Conv2DTranspose(32, 3, padding='same', activation=activations.relu, strides=2)(out)
    out = layers.Conv2D(1, 3, padding='same', activation=activations.sigmoid)(out)
    decoder = Model(decode_input_layer, out)
    z_dec = decoder(z)

    class CustomVariationalLayer(layers.Layer):

        def __init__(self):
            super(CustomVariationalLayer, self).__init__()

        def vae_loss(self, x, z_decode_value):
            x = K.flatten(x)
            z_decode_value = K.flatten(z_decode_value)
            xent_loss = binary_crossentropy(x, z_decode_value)
            kl_loss = -5e-4 * K.mean(
                1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1
            )
            return K.mean(xent_loss + kl_loss)

        def call(self, inputs, *args, **kwargs):
            x = inputs[0]
            z_decoded = inputs[1]
            loss = self.vae_loss(x, z_decoded)
            self.add_loss(loss, inputs=inputs)
            return x

    y = CustomVariationalLayer()([input_layer, z_dec])

    vae = Model(input_layer, y)
    vae.compile(optimizer=optimizers.RMSprop(), loss=None)
    vae.summary()

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float) / 255.
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype(np.float) / 255.
    x_test = x_test.reshape(x_test.shape + (1,))

    vae.fit(x_train, None, shuffle=True, epochs=10, batch_size=batch_size, validation_data=(x_test, None))

    n = 15
    digit_img_size = 28
    display_tensor = np.zeros((digit_img_size * n, digit_img_size * n))
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
    for i, xi in enumerate(grid_x):
        for k, yk in enumerate(grid_y):
            z_sample = np.array([[xi, yk]])
            z_sample = np.tile(z_sample, batch_size).reshape((batch_size, 2))
            x_dec = decoder.predict(z_sample, batch_size=batch_size)
            digit = x_dec[0].reshape(digit_img_size, digit_img_size)
            display_tensor[i * digit_img_size:(i+1) * digit_img_size, k * digit_img_size:(k+1) * digit_img_size] = digit
    plt.figure(figsize=(10, 10))
    plt.imshow(display_tensor, cmap='Greys_r')
    plt.show()


def gan_demo():
    import os

    latent_dim = 32
    img_width = 32
    img_height = 32
    img_channel = 3

    generator_input = Input(shape=(latent_dim,))
    out = layers.Dense(128 * 16 * 16)(generator_input)
    out = layers.LeakyReLU()(out)
    out = layers.Reshape((16, 16, 128))(out)
    out = layers.Conv2D(256, 5, padding='same')(out)
    out = layers.LeakyReLU()(out)
    out = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(out)
    out = layers.LeakyReLU()(out)
    out = layers.Conv2D(256, 5, padding='same')(out)
    out = layers.LeakyReLU()(out)
    out = layers.Conv2D(256, 5, padding='same')(out)
    out = layers.LeakyReLU()(out)
    out = layers.Conv2D(img_channel, 7, activation=activations.tanh, padding='same')(out)
    generator = Model(generator_input, out)
    generator.summary()

    discriminator_input = layers.Input(shape=(img_height, img_width, img_channel))
    out = layers.Conv2D(128, 3)(discriminator_input)
    out = layers.LeakyReLU()(out)
    out = layers.Conv2D(128, 4, strides=2)(out)
    out = layers.LeakyReLU()(out)
    out = layers.Conv2D(128, 4, strides=2)(out)
    out = layers.LeakyReLU()(out)
    out = layers.Conv2D(128, 4, strides=2)(out)
    out = layers.LeakyReLU()(out)
    out = layers.Flatten()(out)
    out = layers.Dropout(0.4)(out)
    out = layers.Dense(1, activation=activations.sigmoid)(out)
    discriminator = Model(discriminator_input, out)
    discriminator.summary()
    discriminator.compile(
        optimizer=optimizers.RMSprop(lr=0.0008, clipvalue=1., decay=1e-8),
        loss=losses.BinaryCrossentropy()
    )
    discriminator.trainable = False

    gan_input = Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = Model(gan_input, gan_output)
    gan.summary()
    gan.compile(
        optimizer=optimizers.RMSprop(lr=0.0004, clipvalue=1., decay=1e-8),
        loss=losses.BinaryCrossentropy()
    )

    (x_train, y_train), (_, _) = cifar10.load_data()
    x_train = x_train[y_train.flatten() == 6]
    x_train = x_train.reshape((x_train.shape[0],) + (img_height, img_width, img_channel)).astype(np.float) / 255.

    iter_num = 10000
    batch_size = 20
    save_dir = 'image/gan_out'
    start = 0
    for i in range(iter_num):
        rand_latent = np.random.normal(size=(batch_size, latent_dim))
        generated_img = generator.predict(rand_latent)
        stop = start + batch_size
        real_img = x_train[start:stop]
        combine_img = np.concatenate([generated_img, real_img], axis=0)
        combine_label = np.concatenate([
            np.ones((batch_size, 1)),
            np.zeros((batch_size, 1))
        ], axis=0)
        combine_label += 0.05 * np.random.random(combine_label.shape)
        d_loss = discriminator.train_on_batch(combine_img, combine_label)
        rand_latent = np.random.normal(size=(batch_size, latent_dim))
        mislead_label = np.zeros((batch_size, 1))
        g_loss = gan.train_on_batch(rand_latent, mislead_label)

        start += batch_size
        if start > len(x_train) - batch_size:
            start = 0
        if i % 100 == 0:
            gan.save('dl_model/gan.h5')
            print('step: ', i)
            print('d loss: ', d_loss)
            print('g loss: ', g_loss)
            img = image.array_to_img(generated_img[0] * 255., scale=False)
            img.save(os.path.join(save_dir, 'generated_frog_{}_.jpg'.format(i)))
            img = image.array_to_img(real_img[0] * 255., scale=False)
            img.save(os.path.join(save_dir, 'real_frog_{}_.jpg'.format(i)))

