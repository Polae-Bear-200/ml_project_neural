import tensorflow as tf
from tensorflow.keras.preprocessing import image as kp_image
import numpy as np
import PIL.Image

def load_img(path_to_img):
    max_dim = 512
    img = PIL.Image.open(path_to_img)
    long = max(img.size)
    scale = max_dim / long
    img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), PIL.Image.ANTIALIAS)
    img = kp_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

def preprocess_image(img):
    img = tf.keras.applications.vgg19.preprocess_input(img*255)
    return img

def deprocess_image(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3 
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False

content_layers = ['block5_conv2'] 
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

style_extractor = vgg_layers(style_layers)
content_extractor = vgg_layers(content_layers)

def style_content_model(inputs):
    inputs = preprocess_image(inputs)
    style_outputs = style_extractor(inputs)
    content_outputs = content_extractor(inputs)
    style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
    content_dict = {content_name:value for content_name, value in zip(content_layers, content_outputs)}
    style_dict = {style_name:value for style_name, value in zip(style_layers, style_outputs)}
    return {'content': content_dict, 'style': style_dict}

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

def style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

@tf.function()
def train_step(image, style_targets, content_targets, extractor, style_weight, content_weight):
    with tf.GradientTape() as tape:
        outputs = style_content_model(image)
        loss = style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight)
    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))

def run_style_transfer(content_path, style_path, num_iterations=1000, style_weight=1e-2, content_weight=1e4):
    content_image = load_img(content_path)
    style_image = load_img(style_path)
    content_image = tf.Variable(preprocess_image(content_image), dtype=tf.float32)
    style_image = preprocess_image(style_image)
    style_targets = style_content_model(style_image)['style']
    content_targets = style_content_model(content_image)['content']
    image = tf.Variable(content_image)
    for i in range(num_iterations):
        train_step(image, style_targets, content_targets, style_content_model, style_weight, content_weight)
    final_image = deprocess_image(image.numpy())
    return final_image
