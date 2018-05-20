from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, Input, Lambda, MaxPooling2D, Activation, BatchNormalization, Dropout
from keras.layers.merge import concatenate
from keras.initializers import he_normal

init = he_normal(seed=1)


def conv_BN(x, n_kernel, f_size, s=1):
    x = Conv2D(n_kernel, (f_size, f_size), strides=(s, s), padding='same', kernel_initializer=init)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    return x


def side_out(x, factor):
    x = Conv2D(1, (1, 1), activation=None, padding='same')(x)

    kernel_size = (2*factor, 2*factor)
    x = Conv2DTranspose(1, kernel_size, strides=factor, padding='same',
                        use_bias=False, activation=None, kernel_initializer=init)(x)
    return x


def side_out_2(x, factor):
    x = Conv2D(1, (1, 1), activation=None, padding='same')(x)

    kernel_size = (factor, factor)
    x = Conv2DTranspose(1, kernel_size, strides=factor, padding='same',
                        use_bias=False, activation=None, kernel_initializer=init)(x)
    return x


# Build U-Net model
def u_net_ori(input_shape=None):

    inputs = Input(shape=input_shape)
    # Normalization
    s = Lambda(lambda x: x / 255)(inputs)
    # Block 1
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv_1a', kernel_initializer=init)(s)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv_1b', kernel_initializer=init)(c1)
    p1 = MaxPooling2D((2, 2), name='pool_1')(c1)
    # Block 2
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_2a', kernel_initializer=init)(p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_2b', kernel_initializer=init)(c2)
    p2 = MaxPooling2D((2, 2), name='pool_2')(c2)
    # Block 3
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_3a', kernel_initializer=init)(p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_3b', kernel_initializer=init)(c3)
    p3 = MaxPooling2D((2, 2), name='pool_3')(c3)
    # Block 4
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_4a', kernel_initializer=init)(p3)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_4b', kernel_initializer=init)(c4)
    p4 = MaxPooling2D((2, 2), name='pool_4')(c4)
    # Block 5
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_5a', kernel_initializer=init)(p4)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_5b', kernel_initializer=init)(c5)
    # Block 6
    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='upconv_6', kernel_initializer=init)(c5)
    u6 = concatenate([u6, c4], name='concat_6')
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_6a', kernel_initializer=init)(u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_6b', kernel_initializer=init)(c6)
    # Block 7
    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name='upconv_7', kernel_initializer=init)(c6)
    u7 = concatenate([u7, c3], name='concat_7')
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_7a', kernel_initializer=init)(u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_7b', kernel_initializer=init)(c7)
    # Block 8
    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same', name='upconv_8', kernel_initializer=init)(c7)
    u8 = concatenate([u8, c2], name='concat_8')
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_8a', kernel_initializer=init)(u8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_8b', kernel_initializer=init)(c8)
    # Block 9
    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same', name='upconv_9', kernel_initializer=init)(c8)
    u9 = concatenate([u9, c1], name='concat_9')
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv_9a', kernel_initializer=init)(u9)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv_9b', kernel_initializer=init)(c9)
    # Output
    outputs = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer=init)(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model


#  U-Net + HED model
def u_net_fuse(input_shape=None):

    inputs = Input(shape=input_shape)
    # Normalization
    s = Lambda(lambda x: x / 255)(inputs)
    s = Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer=init)(s)
    s = MaxPooling2D((2, 2))(s)

    # Block 1
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_1a', kernel_initializer=init)(s)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_1b', kernel_initializer=init)(c1)
    p1 = MaxPooling2D((2, 2), name='pool_1')(c1)
    # Block 2
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_2a', kernel_initializer=init)(p1)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_2b', kernel_initializer=init)(c2)
    p2 = MaxPooling2D((2, 2), name='pool_2')(c2)
    # Block 3
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_3a', kernel_initializer=init)(p2)
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_3b', kernel_initializer=init)(c3)
    p3 = MaxPooling2D((2, 2), name='pool_3')(c3)
    # Block 4
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_4a', kernel_initializer=init)(p3)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_4b', kernel_initializer=init)(c4)
    p4 = MaxPooling2D((2, 2), name='pool_4')(c4)

    # Block 5
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_5a', kernel_initializer=init)(p4)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_5b', kernel_initializer=init)(c5)
    s1 = side_out(c5, 32)

    # Block 6
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name='upconv_6', kernel_initializer=init)(c5)
    u6 = concatenate([u6, c4], name='concat_6')
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_6a', kernel_initializer=init)(u6)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_6b', kernel_initializer=init)(c6)
    s2 = side_out(c6, 16)

    # Block 7
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='upconv_7', kernel_initializer=init)(c6)
    u7 = concatenate([u7, c3], name='concat_7')
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_7a', kernel_initializer=init)(u7)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_7b', kernel_initializer=init)(c7)
    s3 = side_out(c7, 8)

    # Block 8
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name='upconv_8', kernel_initializer=init)(c7)
    u8 = concatenate([u8, c2], name='concat_8')
    c8 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_8a', kernel_initializer=init)(u8)
    c8 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_8b', kernel_initializer=init)(c8)
    s4 = side_out(c8, 4)

    # Block 9
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same', name='upconv_9', kernel_initializer=init)(c8)
    u9 = concatenate([u9, c1], name='concat_9')
    c9 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_9a', kernel_initializer=init)(u9)
    c9 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_9b', kernel_initializer=init)(c9)
    s5 = side_out(c9, 2)

    # fuse
    fuse = concatenate(inputs=[s1, s2, s3, s4, s5], axis=-1)
    fuse = Conv2D(1, (1, 1), padding='same', activation=None)(fuse)       # 320 480 1

    # outputs
    o1    = Activation('sigmoid', name='o1')(s1)
    o2    = Activation('sigmoid', name='o2')(s2)
    o3    = Activation('sigmoid', name='o3')(s3)
    o4    = Activation('sigmoid', name='o4')(s4)
    o5    = Activation('sigmoid', name='o5')(s5)
    ofuse = Activation('sigmoid', name='ofuse')(fuse)

    model = Model(inputs=[inputs], outputs=[o1, o2, o3, o4, o5, ofuse])

    return model


#  U-Net + HED model Version 2
def u_net_fuse_v1(input_shape=None):

    inputs = Input(shape=input_shape)
    # Normalization
    s = Lambda(lambda x: x / 255)(inputs)
    s = Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer=init)(s)
    s = MaxPooling2D((2, 2))(s)

    # Block 1
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_1a', kernel_initializer=init)(s)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_1b', kernel_initializer=init)(c1)
    p1 = MaxPooling2D((2, 2), name='pool_1')(c1)
    # Block 2
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_2a', kernel_initializer=init)(p1)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_2b', kernel_initializer=init)(c2)
    p2 = MaxPooling2D((2, 2), name='pool_2')(c2)
    # Block 3
    c3 = Conv2D(64, (5, 5), activation='relu', padding='same', name='conv_3a', kernel_initializer=init)(p2)
    c3 = Conv2D(64, (5, 5), activation='relu', padding='same', name='conv_3b', kernel_initializer=init)(c3)
    p3 = MaxPooling2D((2, 2), name='pool_3')(c3)
    # Block 4
    c4 = Conv2D(128, (7, 7), activation='relu', padding='same', name='conv_4a', kernel_initializer=init)(p3)
    c4 = Conv2D(128, (7, 7), activation='relu', padding='same', name='conv_4b', kernel_initializer=init)(c4)
    p4 = MaxPooling2D((2, 2), name='pool_4')(c4)

    # Block 5
    c5 = Conv2D(256, (11, 11), activation='relu', padding='same', name='conv_5a', kernel_initializer=init)(p4)
    c5 = Conv2D(256, (11, 11), activation='relu', padding='same', name='conv_5b', kernel_initializer=init)(c5)
    s1 = side_out(c5, 32)

    # Block 6
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name='upconv_6', kernel_initializer=init)(c5)
    u6 = concatenate([u6, c4], name='concat_6')
    c6 = Conv2D(128, (7, 7), activation='relu', padding='same', name='conv_6a', kernel_initializer=init)(u6)
    c6 = Conv2D(128, (7, 7), activation='relu', padding='same', name='conv_6b', kernel_initializer=init)(c6)
    s2 = side_out(c6, 16)

    # Block 7
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='upconv_7', kernel_initializer=init)(c6)
    u7 = concatenate([u7, c3], name='concat_7')
    c7 = Conv2D(64, (5, 5), activation='relu', padding='same', name='conv_7a', kernel_initializer=init)(u7)
    c7 = Conv2D(64, (5, 5), activation='relu', padding='same', name='conv_7b', kernel_initializer=init)(c7)
    s3 = side_out(c7, 8)

    # Block 8
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name='upconv_8', kernel_initializer=init)(c7)
    u8 = concatenate([u8, c2], name='concat_8')
    c8 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_8a', kernel_initializer=init)(u8)
    c8 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_8b', kernel_initializer=init)(c8)
    s4 = side_out(c8, 4)

    # Block 9
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same', name='upconv_9', kernel_initializer=init)(c8)
    u9 = concatenate([u9, c1], name='concat_9')
    c9 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_9a', kernel_initializer=init)(u9)
    c9 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_9b', kernel_initializer=init)(c9)
    s5 = side_out(c9, 2)

    # fuse
    fuse = concatenate(inputs=[s1, s2, s3, s4, s5], axis=-1)
    fuse = Conv2D(1, (1, 1), padding='same', activation=None)(fuse)       # 320 480 1

    # outputs
    o1    = Activation('sigmoid', name='o1')(s1)
    o2    = Activation('sigmoid', name='o2')(s2)
    o3    = Activation('sigmoid', name='o3')(s3)
    o4    = Activation('sigmoid', name='o4')(s4)
    o5    = Activation('sigmoid', name='o5')(s5)
    ofuse = Activation('sigmoid', name='ofuse')(fuse)

    model = Model(inputs=[inputs], outputs=[o1, o2, o3, o4, o5, ofuse])

    return model


#  U-Net + HED + BN + Dropout model Version 2
def u_net_fuse_v2(input_shape=None):

    inputs = Input(shape=input_shape)
    # Normalization
    s = Lambda(lambda x: x / 255)(inputs)
    s = Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer=init)(s)
    s = MaxPooling2D((2, 2))(s)

    # Block 1
    c1 = conv_BN(s, n_kernel=16, f_size=3, s=1)
    c1 = conv_BN(c1, n_kernel=16, f_size=3, s=1)
    p1 = MaxPooling2D((2, 2), name='pool_1')(c1)
    d1 = Dropout(0.2)(p1)
    # Block 2
    c2 = conv_BN(d1, n_kernel=32, f_size=3, s=1)
    c2 = conv_BN(c2, n_kernel=32, f_size=3, s=1)
    p2 = MaxPooling2D((2, 2), name='pool_2')(c2)
    d2 = Dropout(0.2)(p2)
    # Block 3
    c3 = conv_BN(d2, n_kernel=64, f_size=3, s=1)
    c3 = conv_BN(c3, n_kernel=64, f_size=3, s=1)
    p3 = MaxPooling2D((2, 2), name='pool_3')(c3)
    d3 = Dropout(0.2)(p3)
    # Block 4
    c4 = conv_BN(d3, n_kernel=128, f_size=3, s=1)
    c4 = conv_BN(c4, n_kernel=128, f_size=3, s=1)
    p4 = MaxPooling2D((2, 2), name='pool_4')(c4)
    d4 = Dropout(0.2)(p4)
    # Block 5
    c5 = conv_BN(d4, n_kernel=256, f_size=3, s=1)
    c5 = conv_BN(c5, n_kernel=256, f_size=3, s=1)
    s1 = side_out(c5, 32)

    # Block 6
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name='upconv_6', kernel_initializer=init)(c5)
    u6 = concatenate([u6, c4], name='concat_6')
    c6 = conv_BN(u6, n_kernel=128, f_size=3, s=1)
    c6 = conv_BN(c6, n_kernel=128, f_size=3, s=1)
    s2 = side_out(c6, 16)

    # Block 7
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='upconv_7', kernel_initializer=init)(c6)
    u7 = concatenate([u7, c3], name='concat_7')
    c7 = conv_BN(u7, n_kernel=64, f_size=3, s=1)
    c7 = conv_BN(c7, n_kernel=64, f_size=3, s=1)
    s3 = side_out(c7, 8)

    # Block 8
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name='upconv_8', kernel_initializer=init)(c7)
    u8 = concatenate([u8, c2], name='concat_8')
    c8 = conv_BN(u8, n_kernel=32, f_size=3, s=1)
    c8 = conv_BN(c8, n_kernel=32, f_size=3, s=1)
    s4 = side_out(c8, 4)

    # Block 9
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same', name='upconv_9', kernel_initializer=init)(c8)
    u9 = concatenate([u9, c1], name='concat_9')
    c9 = conv_BN(u9, n_kernel=16, f_size=3, s=1)
    c9 = conv_BN(c9, n_kernel=16, f_size=3, s=1)
    s5 = side_out(c9, 2)

    # fuse
    fuse = concatenate(inputs=[s1, s2, s3, s4, s5], axis=-1)
    fuse = Conv2D(1, (1, 1), padding='same', activation=None)(fuse)       # 320 480 1

    # outputs
    o1    = Activation('sigmoid', name='o1')(s1)
    o2    = Activation('sigmoid', name='o2')(s2)
    o3    = Activation('sigmoid', name='o3')(s3)
    o4    = Activation('sigmoid', name='o4')(s4)
    o5    = Activation('sigmoid', name='o5')(s5)
    ofuse = Activation('sigmoid', name='ofuse')(fuse)

    model = Model(inputs=[inputs], outputs=[o1, o2, o3, o4, o5, ofuse])

    return model


#  U-Net + HED + BN + Dropout + RF model Version 3
def u_net_fuse_v3(input_shape=None):

    inputs = Input(shape=input_shape)
    # Normalization
    s = Lambda(lambda x: x / 255)(inputs)
    s = conv_BN(s, n_kernel=32, f_size=5, s=1)
    s = MaxPooling2D((2, 2))(s)

    # Block 1
    c1 = conv_BN(s, n_kernel=32, f_size=3, s=1)
    c1 = conv_BN(c1, n_kernel=32, f_size=3, s=1)
    p1 = MaxPooling2D((2, 2), name='pool_1')(c1)
    d1 = Dropout(0.2)(p1)
    # Block 2
    c2 = conv_BN(d1, n_kernel=64, f_size=3, s=1)
    c2 = conv_BN(c2, n_kernel=64, f_size=3, s=1)
    p2 = MaxPooling2D((2, 2), name='pool_2')(c2)
    d2 = Dropout(0.2)(p2)
    # Block 3
    c3 = conv_BN(d2, n_kernel=128, f_size=3, s=1)
    c3 = conv_BN(c3, n_kernel=128, f_size=3, s=1)
    p3 = MaxPooling2D((2, 2), name='pool_3')(c3)
    d3 = Dropout(0.2)(p3)
    # Block 4
    c4 = conv_BN(d3, n_kernel=256, f_size=3, s=1)
    c4 = conv_BN(c4, n_kernel=256, f_size=3, s=1)
    p4 = MaxPooling2D((2, 2), name='pool_4')(c4)
    d4 = Dropout(0.2)(p4)
    # Block 5
    c5 = conv_BN(d4, n_kernel=512, f_size=3, s=1)
    c5 = conv_BN(c5, n_kernel=512, f_size=3, s=1)
    s1 = side_out(c5, 32)

    # Block 6
    u6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', name='upconv_6', kernel_initializer=init)(c5)
    u6 = concatenate([u6, c4], name='concat_6')
    c6 = conv_BN(u6, n_kernel=256, f_size=3, s=1)
    c6 = conv_BN(c6, n_kernel=256, f_size=3, s=1)
    s2 = side_out(c6, 16)

    # Block 7
    u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name='upconv_7', kernel_initializer=init)(c6)
    u7 = concatenate([u7, c3], name='concat_7')
    c7 = conv_BN(u7, n_kernel=128, f_size=3, s=1)
    c7 = conv_BN(c7, n_kernel=128, f_size=3, s=1)
    s3 = side_out(c7, 8)

    # Block 8
    u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='upconv_8', kernel_initializer=init)(c7)
    u8 = concatenate([u8, c2], name='concat_8')
    c8 = conv_BN(u8, n_kernel=64, f_size=3, s=1)
    c8 = conv_BN(c8, n_kernel=64, f_size=3, s=1)
    s4 = side_out(c8, 4)

    # Block 9
    u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name='upconv_9', kernel_initializer=init)(c8)
    u9 = concatenate([u9, c1], name='concat_9')
    c9 = conv_BN(u9, n_kernel=32, f_size=3, s=1)
    c9 = conv_BN(c9, n_kernel=32, f_size=3, s=1)
    s5 = side_out(c9, 2)

    # fuse
    fuse = concatenate(inputs=[s1, s2, s3, s4, s5], axis=-1)
    fuse = Conv2D(1, (1, 1), padding='same', activation=None)(fuse)       # 320 480 1

    # outputs
    o1    = Activation('sigmoid', name='o1')(s1)
    o2    = Activation('sigmoid', name='o2')(s2)
    o3    = Activation('sigmoid', name='o3')(s3)
    o4    = Activation('sigmoid', name='o4')(s4)
    o5    = Activation('sigmoid', name='o5')(s5)
    ofuse = Activation('sigmoid', name='ofuse')(fuse)

    model = Model(inputs=[inputs], outputs=[o1, o2, o3, o4, o5, ofuse])

    return model


def u_net_fuse_v4(input_shape=None):

    inputs = Input(shape=input_shape)
    # Normalization
    s = Lambda(lambda x: x / 255)(inputs)
    s = conv_BN(s, n_kernel=64, f_size=5, s=1)
    s = MaxPooling2D((2, 2))(s)

    # Block 1
    c1 = conv_BN(s, n_kernel=64, f_size=3, s=1)
    c1 = conv_BN(c1, n_kernel=64, f_size=3, s=1)
    p1 = MaxPooling2D((2, 2), name='pool_1')(c1)
    d1 = Dropout(0.2)(p1)
    # Block 2
    c2 = conv_BN(d1, n_kernel=128, f_size=3, s=1)
    c2 = conv_BN(c2, n_kernel=128, f_size=3, s=1)
    p2 = MaxPooling2D((2, 2), name='pool_2')(c2)
    d2 = Dropout(0.2)(p2)
    # Block 3
    c3 = conv_BN(d2, n_kernel=256, f_size=3, s=1)
    c3 = conv_BN(c3, n_kernel=256, f_size=3, s=1)
    p3 = MaxPooling2D((2, 2), name='pool_3')(c3)
    d3 = Dropout(0.2)(p3)
    # Block 4
    c4 = conv_BN(d3, n_kernel=512, f_size=3, s=1)
    c4 = conv_BN(c4, n_kernel=512, f_size=3, s=1)
    p4 = MaxPooling2D((2, 2), name='pool_4')(c4)
    d4 = Dropout(0.2)(p4)
    # Block 5
    c5 = conv_BN(d4, n_kernel=1024, f_size=3, s=1)
    c5 = conv_BN(c5, n_kernel=1024, f_size=3, s=1)
    s1 = side_out(c5, 32)

    # Block 6
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same', name='upconv_6', kernel_initializer=init)(c5)
    u6 = concatenate([u6, c4], name='concat_6')
    c6 = conv_BN(u6, n_kernel=512, f_size=3, s=1)
    c6 = conv_BN(c6, n_kernel=512, f_size=3, s=1)
    s2 = side_out(c6, 16)

    # Block 7
    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', name='upconv_7', kernel_initializer=init)(c6)
    u7 = concatenate([u7, c3], name='concat_7')
    c7 = conv_BN(u7, n_kernel=256, f_size=3, s=1)
    c7 = conv_BN(c7, n_kernel=256, f_size=3, s=1)
    s3 = side_out(c7, 8)

    # Block 8
    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name='upconv_8', kernel_initializer=init)(c7)
    u8 = concatenate([u8, c2], name='concat_8')
    c8 = conv_BN(u8, n_kernel=128, f_size=3, s=1)
    c8 = conv_BN(c8, n_kernel=128, f_size=3, s=1)
    s4 = side_out(c8, 4)

    # Block 9
    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='upconv_9', kernel_initializer=init)(c8)
    u9 = concatenate([u9, c1], name='concat_9')
    c9 = conv_BN(u9, n_kernel=64, f_size=3, s=1)
    c9 = conv_BN(c9, n_kernel=64, f_size=3, s=1)
    s5 = side_out(c9, 2)

    # fuse
    fuse = concatenate(inputs=[s1, s2, s3, s4, s5], axis=-1)
    fuse = Conv2D(1, (1, 1), padding='same', activation=None)(fuse)       # 320 480 1

    # outputs
    o1    = Activation('sigmoid', name='o1')(s1)
    o2    = Activation('sigmoid', name='o2')(s2)
    o3    = Activation('sigmoid', name='o3')(s3)
    o4    = Activation('sigmoid', name='o4')(s4)
    o5    = Activation('sigmoid', name='o5')(s5)
    ofuse = Activation('sigmoid', name='ofuse')(fuse)

    model = Model(inputs=[inputs], outputs=[o1, o2, o3, o4, o5, ofuse])

    return model


def u_net_fuse_v5(input_shape=None):

    inputs = Input(shape=input_shape)
    # Normalization
    s = Lambda(lambda x: x / 255)(inputs)
    s = conv_BN(s, n_kernel=32, f_size=5, s=1)
    s = MaxPooling2D((2, 2))(s)

    # Block 1
    c1 = conv_BN(s, n_kernel=32, f_size=3, s=1)
    c1 = conv_BN(c1, n_kernel=32, f_size=3, s=1)
    p1 = MaxPooling2D((2, 2), name='pool_1')(c1)
    d1 = Dropout(0.2)(p1)
    # Block 2
    c2 = conv_BN(d1, n_kernel=64, f_size=3, s=1)
    c2 = conv_BN(c2, n_kernel=64, f_size=3, s=1)
    p2 = MaxPooling2D((2, 2), name='pool_2')(c2)
    d2 = Dropout(0.2)(p2)
    # Block 3
    c3 = conv_BN(d2, n_kernel=128, f_size=3, s=1)
    c3 = conv_BN(c3, n_kernel=128, f_size=3, s=1)
    p3 = MaxPooling2D((2, 2), name='pool_3')(c3)
    d3 = Dropout(0.2)(p3)
    # Block 4
    c4 = conv_BN(d3, n_kernel=256, f_size=3, s=1)
    c4 = conv_BN(c4, n_kernel=256, f_size=3, s=1)
    p4 = MaxPooling2D((2, 2), name='pool_4')(c4)
    d4 = Dropout(0.2)(p4)
    # Block 5
    c5 = conv_BN(d4, n_kernel=512, f_size=3, s=1)
    c5 = conv_BN(c5, n_kernel=512, f_size=3, s=1)


    # Block 6
    u6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', name='upconv_6', kernel_initializer=init)(c5)
    u6 = concatenate([u6, c4], name='concat_6')
    c6 = conv_BN(u6, n_kernel=256, f_size=3, s=1)
    c6 = conv_BN(c6, n_kernel=256, f_size=3, s=1)
    s1 = side_out(c6, 16)

    # Block 7
    u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name='upconv_7', kernel_initializer=init)(c6)
    u7 = concatenate([u7, c3], name='concat_7')
    c7 = conv_BN(u7, n_kernel=128, f_size=3, s=1)
    c7 = conv_BN(c7, n_kernel=128, f_size=3, s=1)
    s2 = side_out(c7, 8)

    # Block 8
    u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='upconv_8', kernel_initializer=init)(c7)
    u8 = concatenate([u8, c2], name='concat_8')
    c8 = conv_BN(u8, n_kernel=64, f_size=3, s=1)
    c8 = conv_BN(c8, n_kernel=64, f_size=3, s=1)
    s3 = side_out(c8, 4)

    # Block 9
    u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name='upconv_9', kernel_initializer=init)(c8)
    u9 = concatenate([u9, c1], name='concat_9')
    c9 = conv_BN(u9, n_kernel=32, f_size=3, s=1)
    c9 = conv_BN(c9, n_kernel=32, f_size=3, s=1)
    s4 = side_out(c9, 2)

    # fuse
    fuse = concatenate(inputs=[s1, s2, s3, s4], axis=-1)
    fuse = Conv2D(1, (1, 1), padding='same', activation=None)(fuse)       # 320 480 1

    # outputs
    o1    = Activation('sigmoid', name='o1')(s1)
    o2    = Activation('sigmoid', name='o2')(s2)
    o3    = Activation('sigmoid', name='o3')(s3)
    o4    = Activation('sigmoid', name='o4')(s4)
    ofuse = Activation('sigmoid', name='ofuse')(fuse)

    model = Model(inputs=[inputs], outputs=[o1, o2, o3, o4, ofuse])

    return model


def u_net_fuse_v6(input_shape=None):

    inputs = Input(shape=input_shape)
    # Normalization
    s = Lambda(lambda x: x / 255)(inputs)
    s = conv_BN(s, n_kernel=32, f_size=5, s=1)      # 512
    s = MaxPooling2D((2, 2))(s)

    # Block 1
    c1 = conv_BN(s, n_kernel=32, f_size=3, s=1)     # 256
    c1 = conv_BN(c1, n_kernel=32, f_size=3, s=1)
    p1 = MaxPooling2D((2, 2))(c1)
    d1 = Dropout(0.2)(p1)
    # Block 2
    c2 = conv_BN(d1, n_kernel=64, f_size=3, s=1)    # 128
    c2 = conv_BN(c2, n_kernel=64, f_size=3, s=1)
    p2 = MaxPooling2D((2, 2))(c2)
    d2 = Dropout(0.2)(p2)
    # Block 3
    c3 = conv_BN(d2, n_kernel=128, f_size=3, s=1)   # 64
    c3 = conv_BN(c3, n_kernel=128, f_size=3, s=1)
    p3 = MaxPooling2D((2, 2))(c3)
    d3 = Dropout(0.2)(p3)
    # Block 4
    c4 = conv_BN(d3, n_kernel=128, f_size=3, s=1)   # 32
    c4 = conv_BN(c4, n_kernel=128, f_size=3, s=1)
    p4 = MaxPooling2D((2, 2))(c4)
    d4 = Dropout(0.2)(p4)
    # Block 5
    c5 = conv_BN(d4, n_kernel=256, f_size=3, s=1)   # 16
    c5 = conv_BN(c5, n_kernel=256, f_size=3, s=1)
    p5 = MaxPooling2D((2, 2))(c5)
    d5 = Dropout(0.2)(p5)
    # Block 6
    c6 = conv_BN(d5, n_kernel=256, f_size=3, s=1)   # 8
    c6 = conv_BN(c6, n_kernel=256, f_size=3, s=1)
    p6 = MaxPooling2D((2, 2))(c6)
    d6 = Dropout(0.2)(p6)

    # Block 7
    c7 = conv_BN(d6, n_kernel=256, f_size=3, s=1)   # 4
    c7 = conv_BN(c7, n_kernel=256, f_size=3, s=1)

    # Block 8
    u8 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', kernel_initializer=init)(c7)
    u8 = concatenate([u8, c6])
    c8 = conv_BN(u8, n_kernel=256, f_size=3, s=1)
    c8 = conv_BN(c8, n_kernel=256, f_size=3, s=1)

    # Block 9
    u9 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', kernel_initializer=init)(c8)
    u9 = concatenate([u9, c5])
    c9 = conv_BN(u9, n_kernel=256, f_size=3, s=1)
    c9 = conv_BN(c9, n_kernel=256, f_size=3, s=1)

    # Block 10
    u10 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_initializer=init)(c9)
    u10 = concatenate([u10, c4])
    c10 = conv_BN(u10, n_kernel=128, f_size=3, s=1)
    c10 = conv_BN(c10, n_kernel=128, f_size=3, s=1)
    s1 = side_out(c10, 16)

    # Block 11
    u11 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_initializer=init)(c10)
    u11 = concatenate([u11, c3])
    c11 = conv_BN(u11, n_kernel=128, f_size=3, s=1)
    c11 = conv_BN(c11, n_kernel=128, f_size=3, s=1)
    s2 = side_out(c11, 8)

    # Block 12
    u12 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_initializer=init)(c11)
    u12 = concatenate([u12, c2])
    c12 = conv_BN(u12, n_kernel=64, f_size=3, s=1)
    c12 = conv_BN(c12, n_kernel=64, f_size=3, s=1)
    s3 = side_out(c12, 4)

    # Block 13
    u13 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', kernel_initializer=init)(c12)
    u13 = concatenate([u13, c1])
    c13 = conv_BN(u13, n_kernel=32, f_size=3, s=1)
    c13 = conv_BN(c13, n_kernel=32, f_size=3, s=1)
    s4 = side_out(c13, 2)

    # fuse
    fuse = concatenate(inputs=[s1, s2, s3, s4], axis=-1)
    fuse = Conv2D(1, (1, 1), padding='same', activation=None)(fuse)       # 320 480 1

    # outputs
    o1    = Activation('sigmoid', name='o1')(s1)
    o2    = Activation('sigmoid', name='o2')(s2)
    o3    = Activation('sigmoid', name='o3')(s3)
    o4    = Activation('sigmoid', name='o4')(s4)
    ofuse = Activation('sigmoid', name='ofuse')(fuse)

    model = Model(inputs=[inputs], outputs=[o1, o2, o3, o4, ofuse])

    return model


def u_net_fuse_v7(input_shape=None):

    inputs = Input(shape=input_shape)
    # Normalization
    s = Lambda(lambda x: x / 255)(inputs)

    # Block 1
    c1 = conv_BN(s, n_kernel=32, f_size=3, s=1)     # 512
    c1 = conv_BN(c1, n_kernel=32, f_size=3, s=1)
    p1 = MaxPooling2D((2, 2))(c1)
    d1 = Dropout(0.2)(p1)
    # Block 2
    c2 = conv_BN(d1, n_kernel=64, f_size=3, s=1)    # 256
    c2 = conv_BN(c2, n_kernel=64, f_size=3, s=1)
    p2 = MaxPooling2D((2, 2))(c2)
    d2 = Dropout(0.2)(p2)
    # Block 3
    c3 = conv_BN(d2, n_kernel=128, f_size=3, s=1)   # 128
    c3 = conv_BN(c3, n_kernel=128, f_size=3, s=1)
    p3 = MaxPooling2D((2, 2))(c3)
    d3 = Dropout(0.2)(p3)
    # Block 4
    c4 = conv_BN(d3, n_kernel=128, f_size=3, s=1)   # 64
    c4 = conv_BN(c4, n_kernel=128, f_size=3, s=1)
    p4 = MaxPooling2D((2, 2))(c4)
    d4 = Dropout(0.2)(p4)
    # Block 5
    c5 = conv_BN(d4, n_kernel=256, f_size=3, s=1)   # 32
    c5 = conv_BN(c5, n_kernel=256, f_size=3, s=1)
    p5 = MaxPooling2D((2, 2))(c5)
    d5 = Dropout(0.2)(p5)
    # Block 6
    c6 = conv_BN(d5, n_kernel=256, f_size=3, s=1)   # 16
    c6 = conv_BN(c6, n_kernel=256, f_size=3, s=1)
    p6 = MaxPooling2D((2, 2))(c6)
    d6 = Dropout(0.2)(p6)

    # Block 7
    c7 = conv_BN(d6, n_kernel=512, f_size=3, s=1)   # 8
    c7 = conv_BN(c7, n_kernel=512, f_size=3, s=1)

    # Block 8
    u8 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', kernel_initializer=init)(c7)
    u8 = concatenate([u8, c6])
    c8 = conv_BN(u8, n_kernel=256, f_size=3, s=1)

    # Block 9
    u9 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', kernel_initializer=init)(c8)
    u9 = concatenate([u9, c5])
    c9 = conv_BN(u9, n_kernel=256, f_size=3, s=1)

    # Block 10
    u10 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_initializer=init)(c9)
    u10 = concatenate([u10, c4])
    c10 = conv_BN(u10, n_kernel=128, f_size=3, s=1)
    s1 = side_out(c10, 8)

    # Block 11
    u11 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_initializer=init)(c10)
    u11 = concatenate([u11, c3])
    c11 = conv_BN(u11, n_kernel=128, f_size=3, s=1)
    s2 = side_out(c11, 4)

    # Block 12
    u12 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_initializer=init)(c11)
    u12 = concatenate([u12, c2])
    c12 = conv_BN(u12, n_kernel=64, f_size=3, s=1)
    s3 = side_out(c12, 2)

    # Block 13
    u13 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', kernel_initializer=init)(c12)
    u13 = concatenate([u13, c1])
    c13 = conv_BN(u13, n_kernel=32, f_size=3, s=1)
    s4 = side_out(c13, 1)

    # fuse
    fuse = concatenate(inputs=[s1, s2, s3, s4], axis=-1)
    fuse = Conv2D(1, (1, 1), padding='same', activation=None)(fuse)       # 320 480 1

    # outputs
    o1    = Activation('sigmoid', name='o1')(s1)
    o2    = Activation('sigmoid', name='o2')(s2)
    o3    = Activation('sigmoid', name='o3')(s3)
    o4    = Activation('sigmoid', name='o4')(s4)
    ofuse = Activation('sigmoid', name='ofuse')(fuse)

    model = Model(inputs=[inputs], outputs=[o1, o2, o3, o4, ofuse])

    return model


def u_net_fuse_v7_2(input_shape=None):

    inputs = Input(shape=input_shape)
    # Normalization
    s = Lambda(lambda x: x / 255)(inputs)

    # Block 1
    c1 = conv_BN(s, n_kernel=64, f_size=3, s=1)     # 512
    c1 = conv_BN(c1, n_kernel=64, f_size=3, s=1)
    p1 = MaxPooling2D((2, 2))(c1)
    d1 = Dropout(0.2)(p1)
    # Block 2
    c2 = conv_BN(d1, n_kernel=64, f_size=3, s=1)    # 256
    c2 = conv_BN(c2, n_kernel=64, f_size=3, s=1)
    p2 = MaxPooling2D((2, 2))(c2)
    d2 = Dropout(0.2)(p2)
    # Block 3
    c3 = conv_BN(d2, n_kernel=128, f_size=3, s=1)   # 128
    c3 = conv_BN(c3, n_kernel=128, f_size=3, s=1)
    p3 = MaxPooling2D((2, 2))(c3)
    d3 = Dropout(0.2)(p3)
    # Block 4
    c4 = conv_BN(d3, n_kernel=128, f_size=3, s=1)   # 64
    c4 = conv_BN(c4, n_kernel=128, f_size=3, s=1)
    p4 = MaxPooling2D((2, 2))(c4)
    d4 = Dropout(0.2)(p4)
    # Block 5
    c5 = conv_BN(d4, n_kernel=256, f_size=3, s=1)   # 32
    c5 = conv_BN(c5, n_kernel=256, f_size=3, s=1)
    p5 = MaxPooling2D((2, 2))(c5)
    d5 = Dropout(0.2)(p5)
    # Block 6
    c6 = conv_BN(d5, n_kernel=256, f_size=3, s=1)   # 16
    c6 = conv_BN(c6, n_kernel=256, f_size=3, s=1)
    p6 = MaxPooling2D((2, 2))(c6)
    d6 = Dropout(0.2)(p6)

    # Block 7
    c7 = conv_BN(d6, n_kernel=512, f_size=3, s=1)   # 8
    c7 = conv_BN(c7, n_kernel=512, f_size=3, s=1)

    # Block 8
    u8 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', kernel_initializer=init)(c7)
    u8 = concatenate([u8, c6])
    c8 = conv_BN(u8, n_kernel=256, f_size=3, s=1)

    # Block 9
    u9 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', kernel_initializer=init)(c8)
    u9 = concatenate([u9, c5])
    c9 = conv_BN(u9, n_kernel=256, f_size=3, s=1)

    # Block 10
    u10 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_initializer=init)(c9)
    u10 = concatenate([u10, c4])
    c10 = conv_BN(u10, n_kernel=128, f_size=3, s=1)
    s1 = side_out_2(c10, 8)

    # Block 11
    u11 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_initializer=init)(c10)
    u11 = concatenate([u11, c3])
    c11 = conv_BN(u11, n_kernel=128, f_size=3, s=1)
    s2 = side_out_2(c11, 4)

    # Block 12
    u12 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_initializer=init)(c11)
    u12 = concatenate([u12, c2])
    c12 = conv_BN(u12, n_kernel=64, f_size=3, s=1)
    s3 = side_out_2(c12, 2)

    # Block 13
    u13 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_initializer=init)(c12)
    u13 = concatenate([u13, c1])
    c13 = conv_BN(u13, n_kernel=64, f_size=3, s=1)
    s4 = side_out_2(c13, 1)

    # fuse
    fuse = concatenate(inputs=[s1, s2, s3, s4], axis=-1)
    fuse = Conv2D(1, (1, 1), padding='same', activation=None)(fuse)       # 320 480 1

    # outputs
    o1    = Activation('sigmoid', name='o1')(s1)
    o2    = Activation('sigmoid', name='o2')(s2)
    o3    = Activation('sigmoid', name='o3')(s3)
    o4    = Activation('sigmoid', name='o4')(s4)
    ofuse = Activation('sigmoid', name='ofuse')(fuse)

    model = Model(inputs=[inputs], outputs=[o1, o2, o3, o4, ofuse])

    return model
