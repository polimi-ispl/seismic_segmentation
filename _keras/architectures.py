from tensorflow.keras import Model
from tensorflow.keras.models import load_model
import tensorflow.keras.layers as L
from .fcn_models import AtrousFCN_Vgg16_16s, AtrousFCN_Resnet50_16s, FCN_Resnet50_32s, FCN_Vgg16_32s


def fcn(patch_shape, num_classes=1, backbone='resnet', atrous=False, last_act='softmax'):
    assert backbone in ["resnet", "vgg"]
    
    if atrous:
        if backbone == "vgg":
            model = AtrousFCN_Vgg16_16s(input_shape=patch_shape, classes=num_classes, last_act=last_act)
        else:
            model = AtrousFCN_Resnet50_16s(input_shape=patch_shape, classes=num_classes, last_act=last_act)
    else:
        if backbone == "vgg":
            model = FCN_Vgg16_32s(input_shape=patch_shape, classes=num_classes, last_act=last_act)
        else:
            model = FCN_Resnet50_32s(input_shape=patch_shape, classes=num_classes, last_act=last_act)
    
    return model


def fcn_load(filename):
    from .fcn_models import _BilinearUpSampling2D
    return load_model(filename, custom_objects={'_BilinearUpSampling2D': _BilinearUpSampling2D})


class MultiDecoder(Model):
    
    def __init__(self, patch_shape, num_classes=1, separate_classes=False, use_edges=False, use_inputs=False, nf=32,
                 act="relu", backbone="xception", name='MultiDecoder', **kwargs):
        super(MultiDecoder, self).__init__(name=name, **kwargs)

        if backbone == "xception":
            E, D = _Xception_encoder, _Xception_decoder
        elif backbone == "multires":
            E, D = _MRUnet_encoder, _MRUnet_decoder
        
        self.input_layer = L.Input(patch_shape, name="InputPatch")
        self.encoder = E(patch_shape, nf=nf, act=act, name="Encoder")
        
        if separate_classes:
            self.segmenter = [D(self.encoder.output_shape[1:], num_classes=1, nf=nf, act=act,
                                last_act="softmax", name="Segmenter_%d" % n)
                              for n in range(num_classes)]
        else:
            self.segmenter = D(self.encoder.output_shape[1:], num_classes=num_classes, nf=nf, act=act,
                               last_act="softmax", name="Segmenter")
        
        if use_edges:
            self.edge_detector = D(self.encoder.output_shape[1:], num_classes=1, nf=nf, act=act,
                                   last_act="sigmoid", name="EdgeDetector")
        else:
            self.edge_detector = None
        
        if use_inputs:
            self.reconstructor = D(self.encoder.output_shape[1:], num_classes=1, nf=nf, act=act,
                                   last_act="tanh", name="Reconstructor")
        else:
            self.reconstructor = None

        self.build(input_shape=self.input_layer.shape)
    
    def call(self, inputs):
        features = self.encoder(inputs)
        outputs = []
        
        if isinstance(self.segmenter, list):
            outputs += [s(features) for s in self.segmenter]
        else:
            outputs.append(self.segmenter(features))
        
        if self.edge_detector is not None:
            outputs.append(self.edge_detector(features))
        if self.reconstructor is not None:
            outputs.append(self.reconstructor(features))
        
        return outputs


def multidecoder(patch_shape, num_classes=1, separate_classes=False, use_edges=False, use_inputs=False, nf=32,
                 act="relu", backbone="xception"):
    if backbone == "xception":
        E, D = _Xception_encoder, _Xception_decoder
    elif backbone == "multires":
        E, D = _MRUnet_encoder, _MRUnet_decoder
    
    img_layer = L.Input(patch_shape, name="InputPatch")
    enc = E(patch_shape, nf=nf, act=act, name="Encoder")
    features = enc(img_layer)
    
    decoders = []
    outputs = []
    
    if backbone == "xception":
        # segmentation decoder(s)
        if separate_classes:
            decoders += [D(enc.output_shape[1:], num_classes=1, nf=nf, act=act,
                           last_act="softmax", name="Segmentation_%d" % n)
                         for n in range(num_classes)]
            outputs += [decoders[n](features) for n in range(num_classes)]
        else:
            decoders.append(D(enc.output_shape[1:], num_classes=num_classes, nf=nf, act=act,
                              last_act="softmax", name="Segmentation"))
            outputs.append(decoders[-1](features))
        
        # edge detector
        if use_edges:
            decoders.append(D(enc.output_shape[1:], num_classes=1, nf=nf, act=act,
                              last_act="sigmoid", name="EdgeDetector"))
            outputs.append(decoders[-1](features))
        
        # data reconstruction
        if use_inputs:
            decoders.append(D(enc.output_shape[1:], num_classes=1, nf=nf, act=act,
                              last_act="tanh", name="Reconstructor"))
            outputs.append(decoders[-1](features))
    
    elif backbone == "multires":
        features_shape_list = [shape[1:] for shape in enc.output_shape]
        
        # segmentation decoder(s)
        if separate_classes:
            decoders += [D(features_shape_list, num_classes=1, nf=nf, act=act,
                           last_act="softmax", name="Segmentation_%d" % n)
                         for n in range(num_classes)]
            outputs += [decoders[n](features) for n in range(num_classes)]
        else:
            decoders.append(D(features_shape_list, num_classes=num_classes, nf=nf, act=act,
                              last_act="softmax", name="Segmentation"))
            outputs.append(decoders[-1](features))
        
        # edge detector
        if use_edges:
            decoders.append(D(features_shape_list, num_classes=1, nf=nf, act=act,
                              last_act="sigmoid", name="EdgeDetector"))
            outputs.append(decoders[-1](features))
        
        # data reconstruction
        if use_inputs:
            decoders.append(D(features_shape_list, num_classes=1, nf=nf, act=act,
                              last_act="tanh", name="Reconstructor"))
            outputs.append(decoders[-1](features))
    
    return Model(img_layer, outputs, name="MultiDecoder_%s" % backbone)


def _Xception_encoder(patch_shape, nf=32, act="relu", name="Xception_encoder"):
    inputs = L.Input(shape=patch_shape)
    
    # Entry block
    x = L.Conv2D(nf, 3, strides=2, padding="same")(inputs)
    x = L.BatchNormalization()(x)
    x = L.Activation(act)(x)
    
    prev_block_act = x  # Set aside residual
    
    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [nf * 2, nf * 4, nf * 8]:
        x = L.Activation(act)(x)
        x = L.SeparableConv2D(filters, 3, padding="same")(x)
        x = L.BatchNormalization()(x)
        
        x = L.Activation(act)(x)
        x = L.SeparableConv2D(filters, 3, padding="same")(x)
        x = L.BatchNormalization()(x)
        
        x = L.MaxPooling2D(3, strides=2, padding="same")(x)
        
        # Project residual
        r = L.Conv2D(filters, 1, strides=2, padding="same")(prev_block_act)
        x = L.add([x, r])  # Add back residual
        prev_block_act = x  # Set aside next residual
    output = x
    model = Model(inputs, output, name=name)
    return model


def _Xception_decoder(latent_shape, num_classes=1, nf=32, act="relu", last_act="softmax",
                      name="Xception_decoder"):
    inputs = L.Input(shape=latent_shape)
    x = inputs
    prev_block_act = x
    for filters in [nf * 8, nf * 4, nf * 2, nf]:
        x = L.Activation(act)(x)
        x = L.Conv2DTranspose(filters, 3, padding="same")(x)
        x = L.BatchNormalization()(x)
        
        x = L.Activation(act)(x)
        x = L.Conv2DTranspose(filters, 3, padding="same")(x)
        x = L.BatchNormalization()(x)
        
        x = L.UpSampling2D(2)(x)
        
        # Project residual
        r = L.UpSampling2D(2)(prev_block_act)
        r = L.Conv2D(filters, 1, padding="same")(r)
        x = L.add([x, r])  # Add back residual
        prev_block_act = x  # Set aside next residual
    
    # Add a per-pixel classification layer
    outputs = L.Conv2D(num_classes, 3, activation=last_act, padding="same")(x)
    
    # Define the model
    model = Model(inputs, outputs, name=name)
    return model


def Unet(patch_shape, nf=32, num_classes=1, act='relu', kernel_init='he_normal', last_act="sigmoid"):
    inputs = L.Input(patch_shape)
    
    conv1 = L.Conv2D(nf * 2 ** 0, 3, activation=act, padding='same', kernel_initializer=kernel_init)(inputs)
    conv1 = L.Conv2D(nf * 2 ** 0, 3, activation=act, padding='same', kernel_initializer=kernel_init)(conv1)
    pool1 = L.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = L.Conv2D(nf * 2 ** 1, 3, activation=act, padding='same', kernel_initializer=kernel_init)(pool1)
    conv2 = L.Conv2D(nf * 2 ** 1, 3, activation=act, padding='same', kernel_initializer=kernel_init)(conv2)
    pool2 = L.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = L.Conv2D(nf * 2 ** 2, 3, activation=act, padding='same', kernel_initializer=kernel_init)(pool2)
    conv3 = L.Conv2D(nf * 2 ** 2, 3, activation=act, padding='same', kernel_initializer=kernel_init)(conv3)
    pool3 = L.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = L.Conv2D(nf * 2 ** 3, 3, activation=act, padding='same', kernel_initializer=kernel_init)(pool3)
    conv4 = L.Conv2D(nf * 2 ** 3, 3, activation=act, padding='same', kernel_initializer=kernel_init)(conv4)
    drop4 = L.Dropout(0.5)(conv4)
    pool4 = L.MaxPooling2D(pool_size=(2, 2))(drop4)
    
    conv5 = L.Conv2D(nf * 2 ** 4, 3, activation=act, padding='same', kernel_initializer=kernel_init)(pool4)
    conv5 = L.Conv2D(nf * 2 ** 4, 3, activation=act, padding='same', kernel_initializer=kernel_init)(conv5)
    drop5 = L.Dropout(0.5)(conv5)
    
    up6 = L.Conv2D(nf * 2 ** 3, 2, activation=act, padding='same', kernel_initializer=kernel_init)(
        L.UpSampling2D(size=(2, 2))(drop5))
    merge6 = L.concatenate([drop4, up6], axis=3)
    conv6 = L.Conv2D(nf * 2 ** 3, 3, activation=act, padding='same', kernel_initializer=kernel_init)(merge6)
    conv6 = L.Conv2D(nf * 2 ** 3, 3, activation=act, padding='same', kernel_initializer=kernel_init)(conv6)
    
    up7 = L.Conv2D(nf * 2 ** 2, 2, activation=act, padding='same', kernel_initializer=kernel_init)(
        L.UpSampling2D(size=(2, 2))(conv6))
    merge7 = L.concatenate([conv3, up7], axis=3)
    conv7 = L.Conv2D(nf * 2 ** 2, 3, activation=act, padding='same', kernel_initializer=kernel_init)(merge7)
    conv7 = L.Conv2D(nf * 2 ** 2, 3, activation=act, padding='same', kernel_initializer=kernel_init)(conv7)
    
    up8 = L.Conv2D(nf * 2 ** 1, 2, activation=act, padding='same', kernel_initializer=kernel_init)(
        L.UpSampling2D(size=(2, 2))(conv7))
    merge8 = L.concatenate([conv2, up8], axis=3)
    conv8 = L.Conv2D(nf * 2 ** 1, 3, activation=act, padding='same', kernel_initializer=kernel_init)(merge8)
    conv8 = L.Conv2D(nf * 2 ** 1, 3, activation=act, padding='same', kernel_initializer=kernel_init)(conv8)
    
    up9 = L.Conv2D(nf * 2 ** 0, 2, activation=act, padding='same', kernel_initializer=kernel_init)(
        L.UpSampling2D(size=(2, 2))(conv8))
    merge9 = L.concatenate([conv1, up9], axis=3)
    conv9 = L.Conv2D(nf * 2 ** 0, 3, activation=act, padding='same', kernel_initializer=kernel_init)(merge9)
    conv9 = L.Conv2D(nf * 2 ** 0, 3, activation=act, padding='same', kernel_initializer=kernel_init)(conv9)
    conv9 = L.Conv2D(2, 3, activation=act, padding='same', kernel_initializer=kernel_init)(conv9)
    conv10 = L.Conv2D(num_classes, 1, activation=last_act)(conv9)
    
    return Model(inputs=inputs, outputs=conv10, name='Unet')


###############################################################################
#               MultiResolution U-Net 2D
###############################################################################
def _conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None, use_bn=True):
    """
    2D Convolutional layers

    Arguments:
        x {keras layer} -- input layer
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters

    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(1, 1)})
        activation {str} -- activation function (default: {'relu'})
        name {str} -- name of the layer (default: {None})

    Returns:
        [keras layer] -- [output layer]
    """
    
    x = L.Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    if use_bn:
        x = L.BatchNormalization(axis=3, scale=False)(x)
    
    if activation is None:
        return x
    
    x = L.Activation(activation, name=name)(x)
    
    return x


def _trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):
    """
    2D Transposed Convolutional layers

    Arguments:
        x {keras layer} -- input layer
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters

    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(2, 2)})
        name {str} -- name of the layer (default: {None})

    Returns:
        [keras layer] -- [output layer]
    """
    
    x = L.Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)
    x = L.BatchNormalization(axis=3, scale=False)(x)
    
    return x


def _MultiResBlock(U, inp, alpha=1.67, act='relu'):
    """
    MultiRes Block

    Arguments:
        U {int} -- Number of filters in a corrsponding UNet stage
        inp {keras layer} -- input layer

    Returns:
        [keras layer] -- [output layer]
    """
    
    W = alpha * U
    
    shortcut = inp
    
    shortcut = _conv2d_bn(shortcut, int(W * 0.167) + int(W * 0.333) +
                          int(W * 0.5), 1, 1, activation=None, padding='same')
    
    conv3x3 = _conv2d_bn(inp, int(W * 0.167), 3, 3,
                         activation=act, padding='same')
    
    conv5x5 = _conv2d_bn(conv3x3, int(W * 0.333), 3, 3,
                         activation=act, padding='same')
    
    conv7x7 = _conv2d_bn(conv5x5, int(W * 0.5), 3, 3,
                         activation=act, padding='same')
    
    out = L.concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    out = L.BatchNormalization(axis=3)(out)
    
    out = L.add([shortcut, out])
    out = L.Activation(act)(out)
    out = L.BatchNormalization(axis=3)(out)
    
    return out


def _ResPath(filters, length, inp, act='relu'):
    """
    ResPath

    Arguments:
        filters {int} -- [description]
        length {int} -- length of ResPath
        inp {keras layer} -- input layer

    Returns:
        [keras layer] -- [output layer]
    """
    
    shortcut = inp
    shortcut = _conv2d_bn(shortcut, filters, 1, 1, activation=None, padding='same')
    
    out = _conv2d_bn(inp, filters, 3, 3, activation=act, padding='same')
    
    out = L.add([shortcut, out])
    out = L.Activation(act)(out)
    out = L.BatchNormalization(axis=3)(out)
    
    for i in range(length - 1):
        shortcut = out
        shortcut = _conv2d_bn(shortcut, filters, 1, 1, activation=None, padding='same')
        
        out = _conv2d_bn(out, filters, 3, 3, activation=act, padding='same')
        
        out = L.add([shortcut, out])
        out = L.Activation(act)(out)
        out = L.BatchNormalization(axis=3)(out)
    
    return out


def MultiResUnet(patch_shape, nf=32, num_classes=1, keep_last_bn=True, last_act="sigmoid"):
    inputs = L.Input(patch_shape)
    
    mresblock1 = _MultiResBlock(nf, inputs)
    pool1 = L.MaxPooling2D(pool_size=(2, 2))(mresblock1)
    mresblock1 = _ResPath(nf, 4, mresblock1)
    
    mresblock2 = _MultiResBlock(nf * 2, pool1)
    pool2 = L.MaxPooling2D(pool_size=(2, 2))(mresblock2)
    mresblock2 = _ResPath(nf * 2, 3, mresblock2)
    
    mresblock3 = _MultiResBlock(nf * 4, pool2)
    pool3 = L.MaxPooling2D(pool_size=(2, 2))(mresblock3)
    mresblock3 = _ResPath(nf * 4, 2, mresblock3)
    
    mresblock4 = _MultiResBlock(nf * 8, pool3)
    pool4 = L.MaxPooling2D(pool_size=(2, 2))(mresblock4)
    mresblock4 = _ResPath(nf * 8, 1, mresblock4)
    
    mresblock5 = _MultiResBlock(nf * 16, pool4)
    
    up6 = L.concatenate([L.Conv2DTranspose(nf * 8, (2, 2), strides=(2, 2), padding='same')(mresblock5),
                         mresblock4], axis=3)
    mresblock6 = _MultiResBlock(nf * 8, up6)
    
    up7 = L.concatenate([L.Conv2DTranspose(nf * 4, (2, 2), strides=(2, 2), padding='same')(mresblock6),
                         mresblock3], axis=3)
    mresblock7 = _MultiResBlock(nf * 4, up7)
    
    up8 = L.concatenate([L.Conv2DTranspose(nf * 2, (2, 2), strides=(2, 2), padding='same')(mresblock7),
                         mresblock2], axis=3)
    mresblock8 = _MultiResBlock(nf * 2, up8)
    
    up9 = L.concatenate([L.Conv2DTranspose(nf, (2, 2), strides=(2, 2), padding='same')(mresblock8),
                         mresblock1], axis=3)
    mresblock9 = _MultiResBlock(nf, up9)
    
    conv10 = _conv2d_bn(mresblock9, num_classes, 1, 1, activation=last_act, use_bn=keep_last_bn)
    
    return Model(inputs=inputs, outputs=conv10, name='MultiResUnet')


def load_multires(filename):
    return load_model(filename, custom_objects={'_MultiResBlock': _MultiResBlock})


def _MRUnet_encoder(patch_shape, nf=32, act="relu", name="MultiResUnet_encoder"):
    inputs = L.Input(patch_shape)
    
    mresblock1 = _MultiResBlock(nf, inputs, act=act)
    pool1 = L.MaxPooling2D(pool_size=(2, 2))(mresblock1)
    mresblock1 = _ResPath(nf, 4, mresblock1, act=act)
    
    mresblock2 = _MultiResBlock(nf * 2, pool1, act=act)
    pool2 = L.MaxPooling2D(pool_size=(2, 2))(mresblock2)
    mresblock2 = _ResPath(nf * 2, 3, mresblock2, act=act)
    
    mresblock3 = _MultiResBlock(nf * 4, pool2, act=act)
    pool3 = L.MaxPooling2D(pool_size=(2, 2))(mresblock3)
    mresblock3 = _ResPath(nf * 4, 2, mresblock3, act=act)
    
    mresblock4 = _MultiResBlock(nf * 8, pool3, act=act)
    pool4 = L.MaxPooling2D(pool_size=(2, 2))(mresblock4)
    mresblock4 = _ResPath(nf * 8, 1, mresblock4, act=act)
    
    mresblock5 = _MultiResBlock(nf * 16, pool4, act=act)
    
    resblocks = [mresblock1, mresblock2, mresblock3, mresblock4, mresblock5]
    return Model(inputs=inputs, outputs=resblocks, name=name)


def _MRUnet_decoder(mresblock_shape_lists, num_classes=1, nf=32, keep_last_bn=True, last_act="sigmoid",
                    act="relu", name="MultiResUnet_decoder"):
    inputs = [L.Input(shape) for shape in mresblock_shape_lists]
    
    mresblock1, mresblock2, mresblock3, mresblock4, mresblock5 = inputs
    
    up6 = L.concatenate([L.Conv2DTranspose(nf * 8, (2, 2), strides=(2, 2), padding='same')(mresblock5),
                         mresblock4], axis=3)
    mresblock6 = _MultiResBlock(nf * 8, up6, act=act)
    
    up7 = L.concatenate([L.Conv2DTranspose(nf * 4, (2, 2), strides=(2, 2), padding='same')(mresblock6),
                         mresblock3], axis=3)
    mresblock7 = _MultiResBlock(nf * 4, up7, act=act)
    
    up8 = L.concatenate([L.Conv2DTranspose(nf * 2, (2, 2), strides=(2, 2), padding='same')(mresblock7),
                         mresblock2], axis=3)
    mresblock8 = _MultiResBlock(nf * 2, up8, act=act)
    
    up9 = L.concatenate([L.Conv2DTranspose(nf, (2, 2), strides=(2, 2), padding='same')(mresblock8),
                         mresblock1], axis=3)
    mresblock9 = _MultiResBlock(nf, up9, act=act)
    
    conv10 = _conv2d_bn(mresblock9, num_classes, 1, 1, activation=last_act, use_bn=keep_last_bn)
    
    return Model(inputs=inputs, outputs=conv10, name=name)


###############################################################################
#               MultiResolution U-Net 3D
###############################################################################
def _conv3d_bn(x, filters, num_row, num_col, num_z, padding='same', strides=(1, 1, 1), activation='relu', name=None):
    """
    3D Convolutional layers

    Arguments:
        x {keras layer} -- input layer
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
        num_z {int} -- length along z axis in filters

    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(1, 1, 1)})
        activation {str} -- activation function (default: {'relu'})
        name {str} -- name of the layer (default: {None})

    Returns:
        [keras layer] -- [output layer]
    """
    
    x = L.Conv3D(filters, (num_row, num_col, num_z), strides=strides, padding=padding, use_bias=False)(x)
    x = L.BatchNormalization(axis=4, scale=False)(x)
    
    if activation is None:
        return x
    
    x = L.Activation(activation, name=name)(x)
    return x


def _trans_conv3d_bn(x, filters, num_row, num_col, num_z, padding='same', strides=(2, 2, 2), name=None):
    """
    2D Transposed Convolutional layers

    Arguments:
        x {keras layer} -- input layer
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
        num_z {int} -- length along z axis in filters

    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(2, 2, 2)})
        name {str} -- name of the layer (default: {None})

    Returns:
        [keras layer] -- [output layer]
    """
    
    x = L.Conv3DTranspose(filters, (num_row, num_col, num_z), strides=strides, padding=padding)(x)
    x = L.BatchNormalization(axis=4, scale=False)(x)
    
    return x


def _MultiResBlock3D(U, inp, alpha=1.67):
    """
    MultiRes Block

    Arguments:
        U {int} -- Number of filters in a corrsponding UNet stage
        inp {keras layer} -- input layer

    Returns:
        [keras layer] -- [output layer]
    """
    
    W = alpha * U
    
    shortcut = inp
    
    shortcut = _conv3d_bn(shortcut, int(W * 0.167) + int(W * 0.333) + int(W * 0.5), 1, 1, 1, activation=None,
                          padding='same')
    
    conv3x3 = _conv3d_bn(inp, int(W * 0.167), 3, 3, 3, activation='relu', padding='same')
    
    conv5x5 = _conv3d_bn(conv3x3, int(W * 0.333), 3, 3, 3, activation='relu', padding='same')
    
    conv7x7 = _conv3d_bn(conv5x5, int(W * 0.5), 3, 3, 3, activation='relu', padding='same')
    
    out = L.concatenate([conv3x3, conv5x5, conv7x7], axis=4)
    out = L.BatchNormalization(axis=4)(out)
    
    out = L.add([shortcut, out])
    out = L.Activation('relu')(out)
    out = L.BatchNormalization(axis=4)(out)
    
    return out


def _ResPath3D(filters, length, inp):
    """
    ResPath

    Arguments:
        filters {int} -- [description]
        length {int} -- length of ResPath
        inp {keras layer} -- input layer

    Returns:
        [keras layer] -- [output layer]
    """
    
    shortcut = inp
    shortcut = _conv3d_bn(shortcut, filters, 1, 1, 1, activation=None, padding='same')
    
    out = _conv3d_bn(inp, filters, 3, 3, 3, activation='relu', padding='same')
    
    out = L.add([shortcut, out])
    out = L.Activation('relu')(out)
    out = L.BatchNormalization(axis=4)(out)
    
    for i in range(length - 1):
        shortcut = out
        shortcut = _conv3d_bn(shortcut, filters, 1, 1, 1, activation=None, padding='same')
        
        out = _conv3d_bn(out, filters, 3, 3, 3, activation='relu', padding='same')
        
        out = L.add([shortcut, out])
        out = L.Activation('relu')(out)
        out = L.BatchNormalization(axis=4)(out)
    
    return out


def MultiResUnet3D(x, y, z, n_channels, num_classes=1, last_act='softmax'):
    """
    MultiResUNet3D

    Arguments:
        height {int} -- height of image
        width {int} -- width of image
        z {int} -- length along z axis
        n_channels {int} -- number of channels in image

    Returns:
        [keras model] -- MultiResUNet3D model
    """
    
    inputs = L.Input((x, y, z, n_channels))
    
    mresblock1 = _MultiResBlock3D(32, inputs)
    pool1 = L.MaxPooling3D(pool_size=(2, 2, 2))(mresblock1)
    mresblock1 = _ResPath3D(32, 4, mresblock1)
    
    mresblock2 = _MultiResBlock3D(32 * 2, pool1)
    pool2 = L.MaxPooling3D(pool_size=(2, 2, 2))(mresblock2)
    mresblock2 = _ResPath3D(32 * 2, 3, mresblock2)
    
    mresblock3 = _MultiResBlock3D(32 * 4, pool2)
    pool3 = L.MaxPooling3D(pool_size=(2, 2, 2))(mresblock3)
    mresblock3 = _ResPath3D(32 * 4, 2, mresblock3)
    
    mresblock4 = _MultiResBlock3D(32 * 8, pool3)
    pool4 = L.MaxPooling3D(pool_size=(2, 2, 2))(mresblock4)
    mresblock4 = _ResPath3D(32 * 8, 1, mresblock4)
    
    mresblock5 = _MultiResBlock3D(32 * 16, pool4)
    
    up6 = L.concatenate(
        [L.Conv3DTranspose(32 * 8, (2, 2, 2), strides=(2, 2, 2), padding='same')(mresblock5), mresblock4],
        axis=4)
    mresblock6 = _MultiResBlock3D(32 * 8, up6)
    
    up7 = L.concatenate(
        [L.Conv3DTranspose(32 * 4, (2, 2, 2), strides=(2, 2, 2), padding='same')(mresblock6), mresblock3],
        axis=4)
    mresblock7 = _MultiResBlock3D(32 * 4, up7)
    
    up8 = L.concatenate(
        [L.Conv3DTranspose(32 * 2, (2, 2, 2), strides=(2, 2, 2), padding='same')(mresblock7), mresblock2],
        axis=4)
    mresblock8 = _MultiResBlock3D(32 * 2, up8)
    
    up9 = L.concatenate([L.Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(mresblock8), mresblock1],
                        axis=4)
    mresblock9 = _MultiResBlock3D(32, up9)
    
    conv10 = _conv3d_bn(mresblock9, num_classes, 1, 1, 1, activation=last_act)
    
    model = Model(inputs=[inputs], outputs=[conv10])
    
    return model


if __name__ == "__main__":
    import os
    from contextlib import redirect_stdout
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    import tensorflow.keras.backend as K
    from tensorflow.keras.utils import plot_model
    
    K.clear_session()
    
    patch_shape = (992, 128, 1)
    num_classes = 6
    
    separate_classes = False
    use_edges = False
    use_inputs = True
    
    # # keras implementation
    # model = multidecoder(patch_shape,
    #                      num_classes=num_classes,
    #                      separate_classes=separate_classes,
    #                      use_edges=use_edges,
    #                      use_inputs=use_inputs,
    #                      nf=32,
    #                      act="relu",
    #                      backbone="xception")
    # plot_model(model,
    #            to_file="multidecoder_xception.png",
    #            show_shapes=False,
    #            show_layer_names=True,
    #            rankdir="TB",
    #            expand_nested=True,
    #            dpi=96)
    
    model = multidecoder(patch_shape,
                         num_classes=num_classes,
                         separate_classes=separate_classes,
                         use_edges=use_edges,
                         use_inputs=use_inputs,
                         nf=32,
                         act="relu",
                         backbone="multires")
    model.summary()
    print(0)
