from paddle.fluid.initializer import Constant
from paddle.fluid.param_attr import ParamAttr
import paddle.fluid as fluid

class Fddensenet():
    def __init__(self):
        pass

    def net(self, x2paddle_input):
        x2paddle_fddensenet161_classifier_bias = fluid.layers.create_parameter(dtype='float32', shape=[121],
                                                                               name='x2paddle_fddensenet161_classifier_bias',
                                                                               attr='x2paddle_fddensenet161_classifier_bias',
                                                                               default_initializer=Constant(0.0))
        x2paddle_fddensenet161_classifier_weight = fluid.layers.create_parameter(dtype='float32', shape=[121, 2208],
                                                                                 name='x2paddle_fddensenet161_classifier_weight',
                                                                                 attr='x2paddle_fddensenet161_classifier_weight',
                                                                                 default_initializer=Constant(0.0))
        x2paddle_968 = fluid.layers.conv2d(x2paddle_input, num_filters=96, filter_size=[7, 7], stride=[2, 2],
                                           padding=[3, 3], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fddensenet161_features_conv0_weight',
                                           name='x2paddle_968', bias_attr=False)
        x2paddle_969 = fluid.layers.batch_norm(x2paddle_968, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fddensenet161_features_norm0_weight',
                                               bias_attr='x2paddle_fddensenet161_features_norm0_bias',
                                               moving_mean_name='x2paddle_fddensenet161_features_norm0_running_mean',
                                               moving_variance_name='x2paddle_fddensenet161_features_norm0_running_var',
                                               use_global_stats=False, name='x2paddle_969')
        x2paddle_970 = fluid.layers.relu(x2paddle_969, name='x2paddle_970')
        x2paddle_971 = fluid.layers.pool2d(x2paddle_970, pool_size=[3, 3], pool_type='max', pool_stride=[2, 2],
                                           pool_padding=[1, 1], ceil_mode=False, name='x2paddle_971', exclusive=False)
        x2paddle_972 = fluid.layers.concat([x2paddle_971], axis=1)
        x2paddle_973 = fluid.layers.batch_norm(x2paddle_972, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fddensenet161_features_denseblock1_denselayer1_norm1_weight',
                                               bias_attr='x2paddle_fddensenet161_features_denseblock1_denselayer1_norm1_bias',
                                               moving_mean_name='x2paddle_fddensenet161_features_denseblock1_denselayer1_norm1_running_mean',
                                               moving_variance_name='x2paddle_fddensenet161_features_denseblock1_denselayer1_norm1_running_var',
                                               use_global_stats=False, name='x2paddle_973')
        x2paddle_974 = fluid.layers.relu(x2paddle_973, name='x2paddle_974')
        x2paddle_975 = fluid.layers.conv2d(x2paddle_974, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fddensenet161_features_denseblock1_denselayer1_conv1_weight',
                                           name='x2paddle_975', bias_attr=False)
        x2paddle_976 = fluid.layers.batch_norm(x2paddle_975, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fddensenet161_features_denseblock1_denselayer1_norm2_weight',
                                               bias_attr='x2paddle_fddensenet161_features_denseblock1_denselayer1_norm2_bias',
                                               moving_mean_name='x2paddle_fddensenet161_features_denseblock1_denselayer1_norm2_running_mean',
                                               moving_variance_name='x2paddle_fddensenet161_features_denseblock1_denselayer1_norm2_running_var',
                                               use_global_stats=False, name='x2paddle_976')
        x2paddle_977 = fluid.layers.relu(x2paddle_976, name='x2paddle_977')
        x2paddle_978 = fluid.layers.conv2d(x2paddle_977, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                           padding=[1, 1], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fddensenet161_features_denseblock1_denselayer1_conv2_weight',
                                           name='x2paddle_978', bias_attr=False)
        x2paddle_979 = fluid.layers.concat([x2paddle_971, x2paddle_978], axis=1)
        x2paddle_980 = fluid.layers.batch_norm(x2paddle_979, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fddensenet161_features_denseblock1_denselayer2_norm1_weight',
                                               bias_attr='x2paddle_fddensenet161_features_denseblock1_denselayer2_norm1_bias',
                                               moving_mean_name='x2paddle_fddensenet161_features_denseblock1_denselayer2_norm1_running_mean',
                                               moving_variance_name='x2paddle_fddensenet161_features_denseblock1_denselayer2_norm1_running_var',
                                               use_global_stats=False, name='x2paddle_980')
        x2paddle_981 = fluid.layers.relu(x2paddle_980, name='x2paddle_981')
        x2paddle_982 = fluid.layers.conv2d(x2paddle_981, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fddensenet161_features_denseblock1_denselayer2_conv1_weight',
                                           name='x2paddle_982', bias_attr=False)
        x2paddle_983 = fluid.layers.batch_norm(x2paddle_982, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fddensenet161_features_denseblock1_denselayer2_norm2_weight',
                                               bias_attr='x2paddle_fddensenet161_features_denseblock1_denselayer2_norm2_bias',
                                               moving_mean_name='x2paddle_fddensenet161_features_denseblock1_denselayer2_norm2_running_mean',
                                               moving_variance_name='x2paddle_fddensenet161_features_denseblock1_denselayer2_norm2_running_var',
                                               use_global_stats=False, name='x2paddle_983')
        x2paddle_984 = fluid.layers.relu(x2paddle_983, name='x2paddle_984')
        x2paddle_985 = fluid.layers.conv2d(x2paddle_984, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                           padding=[1, 1], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fddensenet161_features_denseblock1_denselayer2_conv2_weight',
                                           name='x2paddle_985', bias_attr=False)
        x2paddle_986 = fluid.layers.concat([x2paddle_971, x2paddle_978, x2paddle_985], axis=1)
        x2paddle_987 = fluid.layers.batch_norm(x2paddle_986, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fddensenet161_features_denseblock1_denselayer3_norm1_weight',
                                               bias_attr='x2paddle_fddensenet161_features_denseblock1_denselayer3_norm1_bias',
                                               moving_mean_name='x2paddle_fddensenet161_features_denseblock1_denselayer3_norm1_running_mean',
                                               moving_variance_name='x2paddle_fddensenet161_features_denseblock1_denselayer3_norm1_running_var',
                                               use_global_stats=False, name='x2paddle_987')
        x2paddle_988 = fluid.layers.relu(x2paddle_987, name='x2paddle_988')
        x2paddle_989 = fluid.layers.conv2d(x2paddle_988, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fddensenet161_features_denseblock1_denselayer3_conv1_weight',
                                           name='x2paddle_989', bias_attr=False)
        x2paddle_990 = fluid.layers.batch_norm(x2paddle_989, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fddensenet161_features_denseblock1_denselayer3_norm2_weight',
                                               bias_attr='x2paddle_fddensenet161_features_denseblock1_denselayer3_norm2_bias',
                                               moving_mean_name='x2paddle_fddensenet161_features_denseblock1_denselayer3_norm2_running_mean',
                                               moving_variance_name='x2paddle_fddensenet161_features_denseblock1_denselayer3_norm2_running_var',
                                               use_global_stats=False, name='x2paddle_990')
        x2paddle_991 = fluid.layers.relu(x2paddle_990, name='x2paddle_991')
        x2paddle_992 = fluid.layers.conv2d(x2paddle_991, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                           padding=[1, 1], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fddensenet161_features_denseblock1_denselayer3_conv2_weight',
                                           name='x2paddle_992', bias_attr=False)
        x2paddle_993 = fluid.layers.concat([x2paddle_971, x2paddle_978, x2paddle_985, x2paddle_992], axis=1)
        x2paddle_994 = fluid.layers.batch_norm(x2paddle_993, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fddensenet161_features_denseblock1_denselayer4_norm1_weight',
                                               bias_attr='x2paddle_fddensenet161_features_denseblock1_denselayer4_norm1_bias',
                                               moving_mean_name='x2paddle_fddensenet161_features_denseblock1_denselayer4_norm1_running_mean',
                                               moving_variance_name='x2paddle_fddensenet161_features_denseblock1_denselayer4_norm1_running_var',
                                               use_global_stats=False, name='x2paddle_994')
        x2paddle_995 = fluid.layers.relu(x2paddle_994, name='x2paddle_995')
        x2paddle_996 = fluid.layers.conv2d(x2paddle_995, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fddensenet161_features_denseblock1_denselayer4_conv1_weight',
                                           name='x2paddle_996', bias_attr=False)
        x2paddle_997 = fluid.layers.batch_norm(x2paddle_996, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fddensenet161_features_denseblock1_denselayer4_norm2_weight',
                                               bias_attr='x2paddle_fddensenet161_features_denseblock1_denselayer4_norm2_bias',
                                               moving_mean_name='x2paddle_fddensenet161_features_denseblock1_denselayer4_norm2_running_mean',
                                               moving_variance_name='x2paddle_fddensenet161_features_denseblock1_denselayer4_norm2_running_var',
                                               use_global_stats=False, name='x2paddle_997')
        x2paddle_998 = fluid.layers.relu(x2paddle_997, name='x2paddle_998')
        x2paddle_999 = fluid.layers.conv2d(x2paddle_998, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                           padding=[1, 1], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fddensenet161_features_denseblock1_denselayer4_conv2_weight',
                                           name='x2paddle_999', bias_attr=False)
        x2paddle_1000 = fluid.layers.concat([x2paddle_971, x2paddle_978, x2paddle_985, x2paddle_992, x2paddle_999],
                                            axis=1)
        x2paddle_1001 = fluid.layers.batch_norm(x2paddle_1000, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock1_denselayer5_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock1_denselayer5_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock1_denselayer5_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock1_denselayer5_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1001')
        x2paddle_1002 = fluid.layers.relu(x2paddle_1001, name='x2paddle_1002')
        x2paddle_1003 = fluid.layers.conv2d(x2paddle_1002, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock1_denselayer5_conv1_weight',
                                            name='x2paddle_1003', bias_attr=False)
        x2paddle_1004 = fluid.layers.batch_norm(x2paddle_1003, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock1_denselayer5_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock1_denselayer5_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock1_denselayer5_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock1_denselayer5_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1004')
        x2paddle_1005 = fluid.layers.relu(x2paddle_1004, name='x2paddle_1005')
        x2paddle_1006 = fluid.layers.conv2d(x2paddle_1005, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock1_denselayer5_conv2_weight',
                                            name='x2paddle_1006', bias_attr=False)
        x2paddle_1007 = fluid.layers.concat(
            [x2paddle_971, x2paddle_978, x2paddle_985, x2paddle_992, x2paddle_999, x2paddle_1006], axis=1)
        x2paddle_1008 = fluid.layers.batch_norm(x2paddle_1007, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock1_denselayer6_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock1_denselayer6_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock1_denselayer6_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock1_denselayer6_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1008')
        x2paddle_1009 = fluid.layers.relu(x2paddle_1008, name='x2paddle_1009')
        x2paddle_1010 = fluid.layers.conv2d(x2paddle_1009, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock1_denselayer6_conv1_weight',
                                            name='x2paddle_1010', bias_attr=False)
        x2paddle_1011 = fluid.layers.batch_norm(x2paddle_1010, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock1_denselayer6_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock1_denselayer6_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock1_denselayer6_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock1_denselayer6_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1011')
        x2paddle_1012 = fluid.layers.relu(x2paddle_1011, name='x2paddle_1012')
        x2paddle_1013 = fluid.layers.conv2d(x2paddle_1012, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock1_denselayer6_conv2_weight',
                                            name='x2paddle_1013', bias_attr=False)
        x2paddle_1014 = fluid.layers.concat(
            [x2paddle_971, x2paddle_978, x2paddle_985, x2paddle_992, x2paddle_999, x2paddle_1006, x2paddle_1013],
            axis=1)
        x2paddle_1015 = fluid.layers.batch_norm(x2paddle_1014, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_transition1_norm_weight',
                                                bias_attr='x2paddle_fddensenet161_features_transition1_norm_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_transition1_norm_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_transition1_norm_running_var',
                                                use_global_stats=False, name='x2paddle_1015')
        x2paddle_1016 = fluid.layers.relu(x2paddle_1015, name='x2paddle_1016')
        x2paddle_1017 = fluid.layers.conv2d(x2paddle_1016, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_transition1_conv_weight',
                                            name='x2paddle_1017', bias_attr=False)
        x2paddle_1018 = fluid.layers.pad2d(x2paddle_1017, pad_value=0.0, mode='constant', paddings=[0, 0, 0, 0],
                                           name='x2paddle_1018')
        x2paddle_1019 = fluid.layers.pool2d(x2paddle_1018, pool_size=[2, 2], pool_type='avg', pool_stride=[2, 2],
                                            pool_padding=[0, 0], ceil_mode=False, exclusive=True, name='x2paddle_1019')
        x2paddle_1020 = fluid.layers.concat([x2paddle_1019], axis=1)
        x2paddle_1021 = fluid.layers.batch_norm(x2paddle_1020, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer1_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock2_denselayer1_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock2_denselayer1_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock2_denselayer1_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1021')
        x2paddle_1022 = fluid.layers.relu(x2paddle_1021, name='x2paddle_1022')
        x2paddle_1023 = fluid.layers.conv2d(x2paddle_1022, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer1_conv1_weight',
                                            name='x2paddle_1023', bias_attr=False)
        x2paddle_1024 = fluid.layers.batch_norm(x2paddle_1023, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer1_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock2_denselayer1_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock2_denselayer1_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock2_denselayer1_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1024')
        x2paddle_1025 = fluid.layers.relu(x2paddle_1024, name='x2paddle_1025')
        x2paddle_1026 = fluid.layers.conv2d(x2paddle_1025, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer1_conv2_weight',
                                            name='x2paddle_1026', bias_attr=False)
        x2paddle_1027 = fluid.layers.concat([x2paddle_1019, x2paddle_1026], axis=1)
        x2paddle_1028 = fluid.layers.batch_norm(x2paddle_1027, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer2_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock2_denselayer2_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock2_denselayer2_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock2_denselayer2_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1028')
        x2paddle_1029 = fluid.layers.relu(x2paddle_1028, name='x2paddle_1029')
        x2paddle_1030 = fluid.layers.conv2d(x2paddle_1029, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer2_conv1_weight',
                                            name='x2paddle_1030', bias_attr=False)
        x2paddle_1031 = fluid.layers.batch_norm(x2paddle_1030, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer2_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock2_denselayer2_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock2_denselayer2_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock2_denselayer2_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1031')
        x2paddle_1032 = fluid.layers.relu(x2paddle_1031, name='x2paddle_1032')
        x2paddle_1033 = fluid.layers.conv2d(x2paddle_1032, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer2_conv2_weight',
                                            name='x2paddle_1033', bias_attr=False)
        x2paddle_1034 = fluid.layers.concat([x2paddle_1019, x2paddle_1026, x2paddle_1033], axis=1)
        x2paddle_1035 = fluid.layers.batch_norm(x2paddle_1034, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer3_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock2_denselayer3_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock2_denselayer3_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock2_denselayer3_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1035')
        x2paddle_1036 = fluid.layers.relu(x2paddle_1035, name='x2paddle_1036')
        x2paddle_1037 = fluid.layers.conv2d(x2paddle_1036, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer3_conv1_weight',
                                            name='x2paddle_1037', bias_attr=False)
        x2paddle_1038 = fluid.layers.batch_norm(x2paddle_1037, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer3_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock2_denselayer3_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock2_denselayer3_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock2_denselayer3_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1038')
        x2paddle_1039 = fluid.layers.relu(x2paddle_1038, name='x2paddle_1039')
        x2paddle_1040 = fluid.layers.conv2d(x2paddle_1039, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer3_conv2_weight',
                                            name='x2paddle_1040', bias_attr=False)
        x2paddle_1041 = fluid.layers.concat([x2paddle_1019, x2paddle_1026, x2paddle_1033, x2paddle_1040], axis=1)
        x2paddle_1042 = fluid.layers.batch_norm(x2paddle_1041, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer4_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock2_denselayer4_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock2_denselayer4_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock2_denselayer4_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1042')
        x2paddle_1043 = fluid.layers.relu(x2paddle_1042, name='x2paddle_1043')
        x2paddle_1044 = fluid.layers.conv2d(x2paddle_1043, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer4_conv1_weight',
                                            name='x2paddle_1044', bias_attr=False)
        x2paddle_1045 = fluid.layers.batch_norm(x2paddle_1044, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer4_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock2_denselayer4_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock2_denselayer4_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock2_denselayer4_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1045')
        x2paddle_1046 = fluid.layers.relu(x2paddle_1045, name='x2paddle_1046')
        x2paddle_1047 = fluid.layers.conv2d(x2paddle_1046, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer4_conv2_weight',
                                            name='x2paddle_1047', bias_attr=False)
        x2paddle_1048 = fluid.layers.concat([x2paddle_1019, x2paddle_1026, x2paddle_1033, x2paddle_1040, x2paddle_1047],
                                            axis=1)
        x2paddle_1049 = fluid.layers.batch_norm(x2paddle_1048, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer5_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock2_denselayer5_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock2_denselayer5_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock2_denselayer5_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1049')
        x2paddle_1050 = fluid.layers.relu(x2paddle_1049, name='x2paddle_1050')
        x2paddle_1051 = fluid.layers.conv2d(x2paddle_1050, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer5_conv1_weight',
                                            name='x2paddle_1051', bias_attr=False)
        x2paddle_1052 = fluid.layers.batch_norm(x2paddle_1051, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer5_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock2_denselayer5_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock2_denselayer5_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock2_denselayer5_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1052')
        x2paddle_1053 = fluid.layers.relu(x2paddle_1052, name='x2paddle_1053')
        x2paddle_1054 = fluid.layers.conv2d(x2paddle_1053, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer5_conv2_weight',
                                            name='x2paddle_1054', bias_attr=False)
        x2paddle_1055 = fluid.layers.concat(
            [x2paddle_1019, x2paddle_1026, x2paddle_1033, x2paddle_1040, x2paddle_1047, x2paddle_1054], axis=1)
        x2paddle_1056 = fluid.layers.batch_norm(x2paddle_1055, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer6_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock2_denselayer6_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock2_denselayer6_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock2_denselayer6_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1056')
        x2paddle_1057 = fluid.layers.relu(x2paddle_1056, name='x2paddle_1057')
        x2paddle_1058 = fluid.layers.conv2d(x2paddle_1057, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer6_conv1_weight',
                                            name='x2paddle_1058', bias_attr=False)
        x2paddle_1059 = fluid.layers.batch_norm(x2paddle_1058, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer6_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock2_denselayer6_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock2_denselayer6_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock2_denselayer6_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1059')
        x2paddle_1060 = fluid.layers.relu(x2paddle_1059, name='x2paddle_1060')
        x2paddle_1061 = fluid.layers.conv2d(x2paddle_1060, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer6_conv2_weight',
                                            name='x2paddle_1061', bias_attr=False)
        x2paddle_1062 = fluid.layers.concat(
            [x2paddle_1019, x2paddle_1026, x2paddle_1033, x2paddle_1040, x2paddle_1047, x2paddle_1054, x2paddle_1061],
            axis=1)
        x2paddle_1063 = fluid.layers.batch_norm(x2paddle_1062, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer7_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock2_denselayer7_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock2_denselayer7_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock2_denselayer7_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1063')
        x2paddle_1064 = fluid.layers.relu(x2paddle_1063, name='x2paddle_1064')
        x2paddle_1065 = fluid.layers.conv2d(x2paddle_1064, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer7_conv1_weight',
                                            name='x2paddle_1065', bias_attr=False)
        x2paddle_1066 = fluid.layers.batch_norm(x2paddle_1065, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer7_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock2_denselayer7_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock2_denselayer7_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock2_denselayer7_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1066')
        x2paddle_1067 = fluid.layers.relu(x2paddle_1066, name='x2paddle_1067')
        x2paddle_1068 = fluid.layers.conv2d(x2paddle_1067, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer7_conv2_weight',
                                            name='x2paddle_1068', bias_attr=False)
        x2paddle_1069 = fluid.layers.concat(
            [x2paddle_1019, x2paddle_1026, x2paddle_1033, x2paddle_1040, x2paddle_1047, x2paddle_1054, x2paddle_1061,
             x2paddle_1068], axis=1)
        x2paddle_1070 = fluid.layers.batch_norm(x2paddle_1069, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer8_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock2_denselayer8_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock2_denselayer8_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock2_denselayer8_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1070')
        x2paddle_1071 = fluid.layers.relu(x2paddle_1070, name='x2paddle_1071')
        x2paddle_1072 = fluid.layers.conv2d(x2paddle_1071, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer8_conv1_weight',
                                            name='x2paddle_1072', bias_attr=False)
        x2paddle_1073 = fluid.layers.batch_norm(x2paddle_1072, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer8_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock2_denselayer8_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock2_denselayer8_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock2_denselayer8_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1073')
        x2paddle_1074 = fluid.layers.relu(x2paddle_1073, name='x2paddle_1074')
        x2paddle_1075 = fluid.layers.conv2d(x2paddle_1074, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer8_conv2_weight',
                                            name='x2paddle_1075', bias_attr=False)
        x2paddle_1076 = fluid.layers.concat(
            [x2paddle_1019, x2paddle_1026, x2paddle_1033, x2paddle_1040, x2paddle_1047, x2paddle_1054, x2paddle_1061,
             x2paddle_1068, x2paddle_1075], axis=1)
        x2paddle_1077 = fluid.layers.batch_norm(x2paddle_1076, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer9_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock2_denselayer9_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock2_denselayer9_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock2_denselayer9_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1077')
        x2paddle_1078 = fluid.layers.relu(x2paddle_1077, name='x2paddle_1078')
        x2paddle_1079 = fluid.layers.conv2d(x2paddle_1078, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer9_conv1_weight',
                                            name='x2paddle_1079', bias_attr=False)
        x2paddle_1080 = fluid.layers.batch_norm(x2paddle_1079, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer9_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock2_denselayer9_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock2_denselayer9_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock2_denselayer9_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1080')
        x2paddle_1081 = fluid.layers.relu(x2paddle_1080, name='x2paddle_1081')
        x2paddle_1082 = fluid.layers.conv2d(x2paddle_1081, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer9_conv2_weight',
                                            name='x2paddle_1082', bias_attr=False)
        x2paddle_1083 = fluid.layers.concat(
            [x2paddle_1019, x2paddle_1026, x2paddle_1033, x2paddle_1040, x2paddle_1047, x2paddle_1054, x2paddle_1061,
             x2paddle_1068, x2paddle_1075, x2paddle_1082], axis=1)
        x2paddle_1084 = fluid.layers.batch_norm(x2paddle_1083, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer10_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock2_denselayer10_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock2_denselayer10_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock2_denselayer10_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1084')
        x2paddle_1085 = fluid.layers.relu(x2paddle_1084, name='x2paddle_1085')
        x2paddle_1086 = fluid.layers.conv2d(x2paddle_1085, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer10_conv1_weight',
                                            name='x2paddle_1086', bias_attr=False)
        x2paddle_1087 = fluid.layers.batch_norm(x2paddle_1086, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer10_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock2_denselayer10_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock2_denselayer10_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock2_denselayer10_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1087')
        x2paddle_1088 = fluid.layers.relu(x2paddle_1087, name='x2paddle_1088')
        x2paddle_1089 = fluid.layers.conv2d(x2paddle_1088, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer10_conv2_weight',
                                            name='x2paddle_1089', bias_attr=False)
        x2paddle_1090 = fluid.layers.concat(
            [x2paddle_1019, x2paddle_1026, x2paddle_1033, x2paddle_1040, x2paddle_1047, x2paddle_1054, x2paddle_1061,
             x2paddle_1068, x2paddle_1075, x2paddle_1082, x2paddle_1089], axis=1)
        x2paddle_1091 = fluid.layers.batch_norm(x2paddle_1090, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer11_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock2_denselayer11_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock2_denselayer11_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock2_denselayer11_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1091')
        x2paddle_1092 = fluid.layers.relu(x2paddle_1091, name='x2paddle_1092')
        x2paddle_1093 = fluid.layers.conv2d(x2paddle_1092, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer11_conv1_weight',
                                            name='x2paddle_1093', bias_attr=False)
        x2paddle_1094 = fluid.layers.batch_norm(x2paddle_1093, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer11_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock2_denselayer11_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock2_denselayer11_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock2_denselayer11_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1094')
        x2paddle_1095 = fluid.layers.relu(x2paddle_1094, name='x2paddle_1095')
        x2paddle_1096 = fluid.layers.conv2d(x2paddle_1095, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer11_conv2_weight',
                                            name='x2paddle_1096', bias_attr=False)
        x2paddle_1097 = fluid.layers.concat(
            [x2paddle_1019, x2paddle_1026, x2paddle_1033, x2paddle_1040, x2paddle_1047, x2paddle_1054, x2paddle_1061,
             x2paddle_1068, x2paddle_1075, x2paddle_1082, x2paddle_1089, x2paddle_1096], axis=1)
        x2paddle_1098 = fluid.layers.batch_norm(x2paddle_1097, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer12_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock2_denselayer12_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock2_denselayer12_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock2_denselayer12_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1098')
        x2paddle_1099 = fluid.layers.relu(x2paddle_1098, name='x2paddle_1099')
        x2paddle_1100 = fluid.layers.conv2d(x2paddle_1099, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer12_conv1_weight',
                                            name='x2paddle_1100', bias_attr=False)
        x2paddle_1101 = fluid.layers.batch_norm(x2paddle_1100, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer12_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock2_denselayer12_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock2_denselayer12_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock2_denselayer12_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1101')
        x2paddle_1102 = fluid.layers.relu(x2paddle_1101, name='x2paddle_1102')
        x2paddle_1103 = fluid.layers.conv2d(x2paddle_1102, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock2_denselayer12_conv2_weight',
                                            name='x2paddle_1103', bias_attr=False)
        x2paddle_1104 = fluid.layers.concat(
            [x2paddle_1019, x2paddle_1026, x2paddle_1033, x2paddle_1040, x2paddle_1047, x2paddle_1054, x2paddle_1061,
             x2paddle_1068, x2paddle_1075, x2paddle_1082, x2paddle_1089, x2paddle_1096, x2paddle_1103], axis=1)
        x2paddle_1105 = fluid.layers.batch_norm(x2paddle_1104, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_transition2_norm_weight',
                                                bias_attr='x2paddle_fddensenet161_features_transition2_norm_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_transition2_norm_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_transition2_norm_running_var',
                                                use_global_stats=False, name='x2paddle_1105')
        x2paddle_1106 = fluid.layers.relu(x2paddle_1105, name='x2paddle_1106')
        x2paddle_1107 = fluid.layers.conv2d(x2paddle_1106, num_filters=384, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_transition2_conv_weight',
                                            name='x2paddle_1107', bias_attr=False)
        x2paddle_1108 = fluid.layers.pad2d(x2paddle_1107, pad_value=0.0, mode='constant', paddings=[0, 0, 0, 0],
                                           name='x2paddle_1108')
        x2paddle_1109 = fluid.layers.pool2d(x2paddle_1108, pool_size=[2, 2], pool_type='avg', pool_stride=[2, 2],
                                            pool_padding=[0, 0], ceil_mode=False, exclusive=True, name='x2paddle_1109')
        x2paddle_1110 = fluid.layers.concat([x2paddle_1109], axis=1)
        x2paddle_1111 = fluid.layers.batch_norm(x2paddle_1110, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer1_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer1_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer1_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer1_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1111')
        x2paddle_1112 = fluid.layers.relu(x2paddle_1111, name='x2paddle_1112')
        x2paddle_1113 = fluid.layers.conv2d(x2paddle_1112, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer1_conv1_weight',
                                            name='x2paddle_1113', bias_attr=False)
        x2paddle_1114 = fluid.layers.batch_norm(x2paddle_1113, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer1_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer1_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer1_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer1_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1114')
        x2paddle_1115 = fluid.layers.relu(x2paddle_1114, name='x2paddle_1115')
        x2paddle_1116 = fluid.layers.conv2d(x2paddle_1115, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer1_conv2_weight',
                                            name='x2paddle_1116', bias_attr=False)
        x2paddle_1117 = fluid.layers.concat([x2paddle_1109, x2paddle_1116], axis=1)
        x2paddle_1118 = fluid.layers.batch_norm(x2paddle_1117, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer2_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer2_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer2_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer2_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1118')
        x2paddle_1119 = fluid.layers.relu(x2paddle_1118, name='x2paddle_1119')
        x2paddle_1120 = fluid.layers.conv2d(x2paddle_1119, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer2_conv1_weight',
                                            name='x2paddle_1120', bias_attr=False)
        x2paddle_1121 = fluid.layers.batch_norm(x2paddle_1120, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer2_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer2_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer2_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer2_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1121')
        x2paddle_1122 = fluid.layers.relu(x2paddle_1121, name='x2paddle_1122')
        x2paddle_1123 = fluid.layers.conv2d(x2paddle_1122, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer2_conv2_weight',
                                            name='x2paddle_1123', bias_attr=False)
        x2paddle_1124 = fluid.layers.concat([x2paddle_1109, x2paddle_1116, x2paddle_1123], axis=1)
        x2paddle_1125 = fluid.layers.batch_norm(x2paddle_1124, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer3_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer3_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer3_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer3_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1125')
        x2paddle_1126 = fluid.layers.relu(x2paddle_1125, name='x2paddle_1126')
        x2paddle_1127 = fluid.layers.conv2d(x2paddle_1126, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer3_conv1_weight',
                                            name='x2paddle_1127', bias_attr=False)
        x2paddle_1128 = fluid.layers.batch_norm(x2paddle_1127, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer3_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer3_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer3_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer3_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1128')
        x2paddle_1129 = fluid.layers.relu(x2paddle_1128, name='x2paddle_1129')
        x2paddle_1130 = fluid.layers.conv2d(x2paddle_1129, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer3_conv2_weight',
                                            name='x2paddle_1130', bias_attr=False)
        x2paddle_1131 = fluid.layers.concat([x2paddle_1109, x2paddle_1116, x2paddle_1123, x2paddle_1130], axis=1)
        x2paddle_1132 = fluid.layers.batch_norm(x2paddle_1131, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer4_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer4_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer4_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer4_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1132')
        x2paddle_1133 = fluid.layers.relu(x2paddle_1132, name='x2paddle_1133')
        x2paddle_1134 = fluid.layers.conv2d(x2paddle_1133, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer4_conv1_weight',
                                            name='x2paddle_1134', bias_attr=False)
        x2paddle_1135 = fluid.layers.batch_norm(x2paddle_1134, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer4_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer4_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer4_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer4_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1135')
        x2paddle_1136 = fluid.layers.relu(x2paddle_1135, name='x2paddle_1136')
        x2paddle_1137 = fluid.layers.conv2d(x2paddle_1136, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer4_conv2_weight',
                                            name='x2paddle_1137', bias_attr=False)
        x2paddle_1138 = fluid.layers.concat([x2paddle_1109, x2paddle_1116, x2paddle_1123, x2paddle_1130, x2paddle_1137],
                                            axis=1)
        x2paddle_1139 = fluid.layers.batch_norm(x2paddle_1138, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer5_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer5_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer5_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer5_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1139')
        x2paddle_1140 = fluid.layers.relu(x2paddle_1139, name='x2paddle_1140')
        x2paddle_1141 = fluid.layers.conv2d(x2paddle_1140, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer5_conv1_weight',
                                            name='x2paddle_1141', bias_attr=False)
        x2paddle_1142 = fluid.layers.batch_norm(x2paddle_1141, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer5_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer5_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer5_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer5_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1142')
        x2paddle_1143 = fluid.layers.relu(x2paddle_1142, name='x2paddle_1143')
        x2paddle_1144 = fluid.layers.conv2d(x2paddle_1143, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer5_conv2_weight',
                                            name='x2paddle_1144', bias_attr=False)
        x2paddle_1145 = fluid.layers.concat(
            [x2paddle_1109, x2paddle_1116, x2paddle_1123, x2paddle_1130, x2paddle_1137, x2paddle_1144], axis=1)
        x2paddle_1146 = fluid.layers.batch_norm(x2paddle_1145, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer6_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer6_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer6_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer6_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1146')
        x2paddle_1147 = fluid.layers.relu(x2paddle_1146, name='x2paddle_1147')
        x2paddle_1148 = fluid.layers.conv2d(x2paddle_1147, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer6_conv1_weight',
                                            name='x2paddle_1148', bias_attr=False)
        x2paddle_1149 = fluid.layers.batch_norm(x2paddle_1148, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer6_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer6_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer6_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer6_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1149')
        x2paddle_1150 = fluid.layers.relu(x2paddle_1149, name='x2paddle_1150')
        x2paddle_1151 = fluid.layers.conv2d(x2paddle_1150, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer6_conv2_weight',
                                            name='x2paddle_1151', bias_attr=False)
        x2paddle_1152 = fluid.layers.concat(
            [x2paddle_1109, x2paddle_1116, x2paddle_1123, x2paddle_1130, x2paddle_1137, x2paddle_1144, x2paddle_1151],
            axis=1)
        x2paddle_1153 = fluid.layers.batch_norm(x2paddle_1152, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer7_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer7_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer7_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer7_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1153')
        x2paddle_1154 = fluid.layers.relu(x2paddle_1153, name='x2paddle_1154')
        x2paddle_1155 = fluid.layers.conv2d(x2paddle_1154, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer7_conv1_weight',
                                            name='x2paddle_1155', bias_attr=False)
        x2paddle_1156 = fluid.layers.batch_norm(x2paddle_1155, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer7_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer7_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer7_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer7_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1156')
        x2paddle_1157 = fluid.layers.relu(x2paddle_1156, name='x2paddle_1157')
        x2paddle_1158 = fluid.layers.conv2d(x2paddle_1157, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer7_conv2_weight',
                                            name='x2paddle_1158', bias_attr=False)
        x2paddle_1159 = fluid.layers.concat(
            [x2paddle_1109, x2paddle_1116, x2paddle_1123, x2paddle_1130, x2paddle_1137, x2paddle_1144, x2paddle_1151,
             x2paddle_1158], axis=1)
        x2paddle_1160 = fluid.layers.batch_norm(x2paddle_1159, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer8_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer8_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer8_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer8_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1160')
        x2paddle_1161 = fluid.layers.relu(x2paddle_1160, name='x2paddle_1161')
        x2paddle_1162 = fluid.layers.conv2d(x2paddle_1161, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer8_conv1_weight',
                                            name='x2paddle_1162', bias_attr=False)
        x2paddle_1163 = fluid.layers.batch_norm(x2paddle_1162, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer8_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer8_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer8_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer8_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1163')
        x2paddle_1164 = fluid.layers.relu(x2paddle_1163, name='x2paddle_1164')
        x2paddle_1165 = fluid.layers.conv2d(x2paddle_1164, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer8_conv2_weight',
                                            name='x2paddle_1165', bias_attr=False)
        x2paddle_1166 = fluid.layers.concat(
            [x2paddle_1109, x2paddle_1116, x2paddle_1123, x2paddle_1130, x2paddle_1137, x2paddle_1144, x2paddle_1151,
             x2paddle_1158, x2paddle_1165], axis=1)
        x2paddle_1167 = fluid.layers.batch_norm(x2paddle_1166, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer9_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer9_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer9_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer9_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1167')
        x2paddle_1168 = fluid.layers.relu(x2paddle_1167, name='x2paddle_1168')
        x2paddle_1169 = fluid.layers.conv2d(x2paddle_1168, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer9_conv1_weight',
                                            name='x2paddle_1169', bias_attr=False)
        x2paddle_1170 = fluid.layers.batch_norm(x2paddle_1169, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer9_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer9_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer9_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer9_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1170')
        x2paddle_1171 = fluid.layers.relu(x2paddle_1170, name='x2paddle_1171')
        x2paddle_1172 = fluid.layers.conv2d(x2paddle_1171, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer9_conv2_weight',
                                            name='x2paddle_1172', bias_attr=False)
        x2paddle_1173 = fluid.layers.concat(
            [x2paddle_1109, x2paddle_1116, x2paddle_1123, x2paddle_1130, x2paddle_1137, x2paddle_1144, x2paddle_1151,
             x2paddle_1158, x2paddle_1165, x2paddle_1172], axis=1)
        x2paddle_1174 = fluid.layers.batch_norm(x2paddle_1173, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer10_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer10_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer10_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer10_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1174')
        x2paddle_1175 = fluid.layers.relu(x2paddle_1174, name='x2paddle_1175')
        x2paddle_1176 = fluid.layers.conv2d(x2paddle_1175, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer10_conv1_weight',
                                            name='x2paddle_1176', bias_attr=False)
        x2paddle_1177 = fluid.layers.batch_norm(x2paddle_1176, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer10_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer10_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer10_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer10_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1177')
        x2paddle_1178 = fluid.layers.relu(x2paddle_1177, name='x2paddle_1178')
        x2paddle_1179 = fluid.layers.conv2d(x2paddle_1178, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer10_conv2_weight',
                                            name='x2paddle_1179', bias_attr=False)
        x2paddle_1180 = fluid.layers.concat(
            [x2paddle_1109, x2paddle_1116, x2paddle_1123, x2paddle_1130, x2paddle_1137, x2paddle_1144, x2paddle_1151,
             x2paddle_1158, x2paddle_1165, x2paddle_1172, x2paddle_1179], axis=1)
        x2paddle_1181 = fluid.layers.batch_norm(x2paddle_1180, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer11_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer11_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer11_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer11_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1181')
        x2paddle_1182 = fluid.layers.relu(x2paddle_1181, name='x2paddle_1182')
        x2paddle_1183 = fluid.layers.conv2d(x2paddle_1182, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer11_conv1_weight',
                                            name='x2paddle_1183', bias_attr=False)
        x2paddle_1184 = fluid.layers.batch_norm(x2paddle_1183, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer11_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer11_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer11_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer11_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1184')
        x2paddle_1185 = fluid.layers.relu(x2paddle_1184, name='x2paddle_1185')
        x2paddle_1186 = fluid.layers.conv2d(x2paddle_1185, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer11_conv2_weight',
                                            name='x2paddle_1186', bias_attr=False)
        x2paddle_1187 = fluid.layers.concat(
            [x2paddle_1109, x2paddle_1116, x2paddle_1123, x2paddle_1130, x2paddle_1137, x2paddle_1144, x2paddle_1151,
             x2paddle_1158, x2paddle_1165, x2paddle_1172, x2paddle_1179, x2paddle_1186], axis=1)
        x2paddle_1188 = fluid.layers.batch_norm(x2paddle_1187, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer12_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer12_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer12_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer12_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1188')
        x2paddle_1189 = fluid.layers.relu(x2paddle_1188, name='x2paddle_1189')
        x2paddle_1190 = fluid.layers.conv2d(x2paddle_1189, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer12_conv1_weight',
                                            name='x2paddle_1190', bias_attr=False)
        x2paddle_1191 = fluid.layers.batch_norm(x2paddle_1190, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer12_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer12_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer12_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer12_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1191')
        x2paddle_1192 = fluid.layers.relu(x2paddle_1191, name='x2paddle_1192')
        x2paddle_1193 = fluid.layers.conv2d(x2paddle_1192, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer12_conv2_weight',
                                            name='x2paddle_1193', bias_attr=False)
        x2paddle_1194 = fluid.layers.concat(
            [x2paddle_1109, x2paddle_1116, x2paddle_1123, x2paddle_1130, x2paddle_1137, x2paddle_1144, x2paddle_1151,
             x2paddle_1158, x2paddle_1165, x2paddle_1172, x2paddle_1179, x2paddle_1186, x2paddle_1193], axis=1)
        x2paddle_1195 = fluid.layers.batch_norm(x2paddle_1194, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer13_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer13_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer13_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer13_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1195')
        x2paddle_1196 = fluid.layers.relu(x2paddle_1195, name='x2paddle_1196')
        x2paddle_1197 = fluid.layers.conv2d(x2paddle_1196, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer13_conv1_weight',
                                            name='x2paddle_1197', bias_attr=False)
        x2paddle_1198 = fluid.layers.batch_norm(x2paddle_1197, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer13_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer13_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer13_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer13_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1198')
        x2paddle_1199 = fluid.layers.relu(x2paddle_1198, name='x2paddle_1199')
        x2paddle_1200 = fluid.layers.conv2d(x2paddle_1199, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer13_conv2_weight',
                                            name='x2paddle_1200', bias_attr=False)
        x2paddle_1201 = fluid.layers.concat(
            [x2paddle_1109, x2paddle_1116, x2paddle_1123, x2paddle_1130, x2paddle_1137, x2paddle_1144, x2paddle_1151,
             x2paddle_1158, x2paddle_1165, x2paddle_1172, x2paddle_1179, x2paddle_1186, x2paddle_1193, x2paddle_1200],
            axis=1)
        x2paddle_1202 = fluid.layers.batch_norm(x2paddle_1201, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer14_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer14_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer14_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer14_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1202')
        x2paddle_1203 = fluid.layers.relu(x2paddle_1202, name='x2paddle_1203')
        x2paddle_1204 = fluid.layers.conv2d(x2paddle_1203, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer14_conv1_weight',
                                            name='x2paddle_1204', bias_attr=False)
        x2paddle_1205 = fluid.layers.batch_norm(x2paddle_1204, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer14_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer14_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer14_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer14_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1205')
        x2paddle_1206 = fluid.layers.relu(x2paddle_1205, name='x2paddle_1206')
        x2paddle_1207 = fluid.layers.conv2d(x2paddle_1206, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer14_conv2_weight',
                                            name='x2paddle_1207', bias_attr=False)
        x2paddle_1208 = fluid.layers.concat(
            [x2paddle_1109, x2paddle_1116, x2paddle_1123, x2paddle_1130, x2paddle_1137, x2paddle_1144, x2paddle_1151,
             x2paddle_1158, x2paddle_1165, x2paddle_1172, x2paddle_1179, x2paddle_1186, x2paddle_1193, x2paddle_1200,
             x2paddle_1207], axis=1)
        x2paddle_1209 = fluid.layers.batch_norm(x2paddle_1208, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer15_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer15_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer15_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer15_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1209')
        x2paddle_1210 = fluid.layers.relu(x2paddle_1209, name='x2paddle_1210')
        x2paddle_1211 = fluid.layers.conv2d(x2paddle_1210, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer15_conv1_weight',
                                            name='x2paddle_1211', bias_attr=False)
        x2paddle_1212 = fluid.layers.batch_norm(x2paddle_1211, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer15_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer15_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer15_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer15_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1212')
        x2paddle_1213 = fluid.layers.relu(x2paddle_1212, name='x2paddle_1213')
        x2paddle_1214 = fluid.layers.conv2d(x2paddle_1213, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer15_conv2_weight',
                                            name='x2paddle_1214', bias_attr=False)
        x2paddle_1215 = fluid.layers.concat(
            [x2paddle_1109, x2paddle_1116, x2paddle_1123, x2paddle_1130, x2paddle_1137, x2paddle_1144, x2paddle_1151,
             x2paddle_1158, x2paddle_1165, x2paddle_1172, x2paddle_1179, x2paddle_1186, x2paddle_1193, x2paddle_1200,
             x2paddle_1207, x2paddle_1214], axis=1)
        x2paddle_1216 = fluid.layers.batch_norm(x2paddle_1215, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer16_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer16_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer16_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer16_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1216')
        x2paddle_1217 = fluid.layers.relu(x2paddle_1216, name='x2paddle_1217')
        x2paddle_1218 = fluid.layers.conv2d(x2paddle_1217, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer16_conv1_weight',
                                            name='x2paddle_1218', bias_attr=False)
        x2paddle_1219 = fluid.layers.batch_norm(x2paddle_1218, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer16_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer16_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer16_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer16_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1219')
        x2paddle_1220 = fluid.layers.relu(x2paddle_1219, name='x2paddle_1220')
        x2paddle_1221 = fluid.layers.conv2d(x2paddle_1220, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer16_conv2_weight',
                                            name='x2paddle_1221', bias_attr=False)
        x2paddle_1222 = fluid.layers.concat(
            [x2paddle_1109, x2paddle_1116, x2paddle_1123, x2paddle_1130, x2paddle_1137, x2paddle_1144, x2paddle_1151,
             x2paddle_1158, x2paddle_1165, x2paddle_1172, x2paddle_1179, x2paddle_1186, x2paddle_1193, x2paddle_1200,
             x2paddle_1207, x2paddle_1214, x2paddle_1221], axis=1)
        x2paddle_1223 = fluid.layers.batch_norm(x2paddle_1222, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer17_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer17_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer17_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer17_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1223')
        x2paddle_1224 = fluid.layers.relu(x2paddle_1223, name='x2paddle_1224')
        x2paddle_1225 = fluid.layers.conv2d(x2paddle_1224, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer17_conv1_weight',
                                            name='x2paddle_1225', bias_attr=False)
        x2paddle_1226 = fluid.layers.batch_norm(x2paddle_1225, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer17_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer17_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer17_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer17_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1226')
        x2paddle_1227 = fluid.layers.relu(x2paddle_1226, name='x2paddle_1227')
        x2paddle_1228 = fluid.layers.conv2d(x2paddle_1227, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer17_conv2_weight',
                                            name='x2paddle_1228', bias_attr=False)
        x2paddle_1229 = fluid.layers.concat(
            [x2paddle_1109, x2paddle_1116, x2paddle_1123, x2paddle_1130, x2paddle_1137, x2paddle_1144, x2paddle_1151,
             x2paddle_1158, x2paddle_1165, x2paddle_1172, x2paddle_1179, x2paddle_1186, x2paddle_1193, x2paddle_1200,
             x2paddle_1207, x2paddle_1214, x2paddle_1221, x2paddle_1228], axis=1)
        x2paddle_1230 = fluid.layers.batch_norm(x2paddle_1229, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer18_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer18_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer18_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer18_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1230')
        x2paddle_1231 = fluid.layers.relu(x2paddle_1230, name='x2paddle_1231')
        x2paddle_1232 = fluid.layers.conv2d(x2paddle_1231, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer18_conv1_weight',
                                            name='x2paddle_1232', bias_attr=False)
        x2paddle_1233 = fluid.layers.batch_norm(x2paddle_1232, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer18_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer18_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer18_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer18_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1233')
        x2paddle_1234 = fluid.layers.relu(x2paddle_1233, name='x2paddle_1234')
        x2paddle_1235 = fluid.layers.conv2d(x2paddle_1234, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer18_conv2_weight',
                                            name='x2paddle_1235', bias_attr=False)
        x2paddle_1236 = fluid.layers.concat(
            [x2paddle_1109, x2paddle_1116, x2paddle_1123, x2paddle_1130, x2paddle_1137, x2paddle_1144, x2paddle_1151,
             x2paddle_1158, x2paddle_1165, x2paddle_1172, x2paddle_1179, x2paddle_1186, x2paddle_1193, x2paddle_1200,
             x2paddle_1207, x2paddle_1214, x2paddle_1221, x2paddle_1228, x2paddle_1235], axis=1)
        x2paddle_1237 = fluid.layers.batch_norm(x2paddle_1236, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer19_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer19_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer19_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer19_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1237')
        x2paddle_1238 = fluid.layers.relu(x2paddle_1237, name='x2paddle_1238')
        x2paddle_1239 = fluid.layers.conv2d(x2paddle_1238, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer19_conv1_weight',
                                            name='x2paddle_1239', bias_attr=False)
        x2paddle_1240 = fluid.layers.batch_norm(x2paddle_1239, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer19_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer19_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer19_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer19_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1240')
        x2paddle_1241 = fluid.layers.relu(x2paddle_1240, name='x2paddle_1241')
        x2paddle_1242 = fluid.layers.conv2d(x2paddle_1241, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer19_conv2_weight',
                                            name='x2paddle_1242', bias_attr=False)
        x2paddle_1243 = fluid.layers.concat(
            [x2paddle_1109, x2paddle_1116, x2paddle_1123, x2paddle_1130, x2paddle_1137, x2paddle_1144, x2paddle_1151,
             x2paddle_1158, x2paddle_1165, x2paddle_1172, x2paddle_1179, x2paddle_1186, x2paddle_1193, x2paddle_1200,
             x2paddle_1207, x2paddle_1214, x2paddle_1221, x2paddle_1228, x2paddle_1235, x2paddle_1242], axis=1)
        x2paddle_1244 = fluid.layers.batch_norm(x2paddle_1243, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer20_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer20_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer20_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer20_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1244')
        x2paddle_1245 = fluid.layers.relu(x2paddle_1244, name='x2paddle_1245')
        x2paddle_1246 = fluid.layers.conv2d(x2paddle_1245, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer20_conv1_weight',
                                            name='x2paddle_1246', bias_attr=False)
        x2paddle_1247 = fluid.layers.batch_norm(x2paddle_1246, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer20_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer20_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer20_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer20_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1247')
        x2paddle_1248 = fluid.layers.relu(x2paddle_1247, name='x2paddle_1248')
        x2paddle_1249 = fluid.layers.conv2d(x2paddle_1248, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer20_conv2_weight',
                                            name='x2paddle_1249', bias_attr=False)
        x2paddle_1250 = fluid.layers.concat(
            [x2paddle_1109, x2paddle_1116, x2paddle_1123, x2paddle_1130, x2paddle_1137, x2paddle_1144, x2paddle_1151,
             x2paddle_1158, x2paddle_1165, x2paddle_1172, x2paddle_1179, x2paddle_1186, x2paddle_1193, x2paddle_1200,
             x2paddle_1207, x2paddle_1214, x2paddle_1221, x2paddle_1228, x2paddle_1235, x2paddle_1242, x2paddle_1249],
            axis=1)
        x2paddle_1251 = fluid.layers.batch_norm(x2paddle_1250, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer21_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer21_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer21_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer21_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1251')
        x2paddle_1252 = fluid.layers.relu(x2paddle_1251, name='x2paddle_1252')
        x2paddle_1253 = fluid.layers.conv2d(x2paddle_1252, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer21_conv1_weight',
                                            name='x2paddle_1253', bias_attr=False)
        x2paddle_1254 = fluid.layers.batch_norm(x2paddle_1253, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer21_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer21_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer21_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer21_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1254')
        x2paddle_1255 = fluid.layers.relu(x2paddle_1254, name='x2paddle_1255')
        x2paddle_1256 = fluid.layers.conv2d(x2paddle_1255, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer21_conv2_weight',
                                            name='x2paddle_1256', bias_attr=False)
        x2paddle_1257 = fluid.layers.concat(
            [x2paddle_1109, x2paddle_1116, x2paddle_1123, x2paddle_1130, x2paddle_1137, x2paddle_1144, x2paddle_1151,
             x2paddle_1158, x2paddle_1165, x2paddle_1172, x2paddle_1179, x2paddle_1186, x2paddle_1193, x2paddle_1200,
             x2paddle_1207, x2paddle_1214, x2paddle_1221, x2paddle_1228, x2paddle_1235, x2paddle_1242, x2paddle_1249,
             x2paddle_1256], axis=1)
        x2paddle_1258 = fluid.layers.batch_norm(x2paddle_1257, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer22_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer22_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer22_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer22_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1258')
        x2paddle_1259 = fluid.layers.relu(x2paddle_1258, name='x2paddle_1259')
        x2paddle_1260 = fluid.layers.conv2d(x2paddle_1259, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer22_conv1_weight',
                                            name='x2paddle_1260', bias_attr=False)
        x2paddle_1261 = fluid.layers.batch_norm(x2paddle_1260, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer22_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer22_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer22_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer22_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1261')
        x2paddle_1262 = fluid.layers.relu(x2paddle_1261, name='x2paddle_1262')
        x2paddle_1263 = fluid.layers.conv2d(x2paddle_1262, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer22_conv2_weight',
                                            name='x2paddle_1263', bias_attr=False)
        x2paddle_1264 = fluid.layers.concat(
            [x2paddle_1109, x2paddle_1116, x2paddle_1123, x2paddle_1130, x2paddle_1137, x2paddle_1144, x2paddle_1151,
             x2paddle_1158, x2paddle_1165, x2paddle_1172, x2paddle_1179, x2paddle_1186, x2paddle_1193, x2paddle_1200,
             x2paddle_1207, x2paddle_1214, x2paddle_1221, x2paddle_1228, x2paddle_1235, x2paddle_1242, x2paddle_1249,
             x2paddle_1256, x2paddle_1263], axis=1)
        x2paddle_1265 = fluid.layers.batch_norm(x2paddle_1264, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer23_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer23_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer23_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer23_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1265')
        x2paddle_1266 = fluid.layers.relu(x2paddle_1265, name='x2paddle_1266')
        x2paddle_1267 = fluid.layers.conv2d(x2paddle_1266, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer23_conv1_weight',
                                            name='x2paddle_1267', bias_attr=False)
        x2paddle_1268 = fluid.layers.batch_norm(x2paddle_1267, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer23_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer23_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer23_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer23_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1268')
        x2paddle_1269 = fluid.layers.relu(x2paddle_1268, name='x2paddle_1269')
        x2paddle_1270 = fluid.layers.conv2d(x2paddle_1269, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer23_conv2_weight',
                                            name='x2paddle_1270', bias_attr=False)
        x2paddle_1271 = fluid.layers.concat(
            [x2paddle_1109, x2paddle_1116, x2paddle_1123, x2paddle_1130, x2paddle_1137, x2paddle_1144, x2paddle_1151,
             x2paddle_1158, x2paddle_1165, x2paddle_1172, x2paddle_1179, x2paddle_1186, x2paddle_1193, x2paddle_1200,
             x2paddle_1207, x2paddle_1214, x2paddle_1221, x2paddle_1228, x2paddle_1235, x2paddle_1242, x2paddle_1249,
             x2paddle_1256, x2paddle_1263, x2paddle_1270], axis=1)
        x2paddle_1272 = fluid.layers.batch_norm(x2paddle_1271, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer24_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer24_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer24_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer24_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1272')
        x2paddle_1273 = fluid.layers.relu(x2paddle_1272, name='x2paddle_1273')
        x2paddle_1274 = fluid.layers.conv2d(x2paddle_1273, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer24_conv1_weight',
                                            name='x2paddle_1274', bias_attr=False)
        x2paddle_1275 = fluid.layers.batch_norm(x2paddle_1274, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer24_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer24_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer24_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer24_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1275')
        x2paddle_1276 = fluid.layers.relu(x2paddle_1275, name='x2paddle_1276')
        x2paddle_1277 = fluid.layers.conv2d(x2paddle_1276, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer24_conv2_weight',
                                            name='x2paddle_1277', bias_attr=False)
        x2paddle_1278 = fluid.layers.concat(
            [x2paddle_1109, x2paddle_1116, x2paddle_1123, x2paddle_1130, x2paddle_1137, x2paddle_1144, x2paddle_1151,
             x2paddle_1158, x2paddle_1165, x2paddle_1172, x2paddle_1179, x2paddle_1186, x2paddle_1193, x2paddle_1200,
             x2paddle_1207, x2paddle_1214, x2paddle_1221, x2paddle_1228, x2paddle_1235, x2paddle_1242, x2paddle_1249,
             x2paddle_1256, x2paddle_1263, x2paddle_1270, x2paddle_1277], axis=1)
        x2paddle_1279 = fluid.layers.batch_norm(x2paddle_1278, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer25_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer25_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer25_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer25_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1279')
        x2paddle_1280 = fluid.layers.relu(x2paddle_1279, name='x2paddle_1280')
        x2paddle_1281 = fluid.layers.conv2d(x2paddle_1280, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer25_conv1_weight',
                                            name='x2paddle_1281', bias_attr=False)
        x2paddle_1282 = fluid.layers.batch_norm(x2paddle_1281, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer25_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer25_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer25_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer25_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1282')
        x2paddle_1283 = fluid.layers.relu(x2paddle_1282, name='x2paddle_1283')
        x2paddle_1284 = fluid.layers.conv2d(x2paddle_1283, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer25_conv2_weight',
                                            name='x2paddle_1284', bias_attr=False)
        x2paddle_1285 = fluid.layers.concat(
            [x2paddle_1109, x2paddle_1116, x2paddle_1123, x2paddle_1130, x2paddle_1137, x2paddle_1144, x2paddle_1151,
             x2paddle_1158, x2paddle_1165, x2paddle_1172, x2paddle_1179, x2paddle_1186, x2paddle_1193, x2paddle_1200,
             x2paddle_1207, x2paddle_1214, x2paddle_1221, x2paddle_1228, x2paddle_1235, x2paddle_1242, x2paddle_1249,
             x2paddle_1256, x2paddle_1263, x2paddle_1270, x2paddle_1277, x2paddle_1284], axis=1)
        x2paddle_1286 = fluid.layers.batch_norm(x2paddle_1285, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer26_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer26_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer26_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer26_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1286')
        x2paddle_1287 = fluid.layers.relu(x2paddle_1286, name='x2paddle_1287')
        x2paddle_1288 = fluid.layers.conv2d(x2paddle_1287, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer26_conv1_weight',
                                            name='x2paddle_1288', bias_attr=False)
        x2paddle_1289 = fluid.layers.batch_norm(x2paddle_1288, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer26_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer26_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer26_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer26_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1289')
        x2paddle_1290 = fluid.layers.relu(x2paddle_1289, name='x2paddle_1290')
        x2paddle_1291 = fluid.layers.conv2d(x2paddle_1290, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer26_conv2_weight',
                                            name='x2paddle_1291', bias_attr=False)
        x2paddle_1292 = fluid.layers.concat(
            [x2paddle_1109, x2paddle_1116, x2paddle_1123, x2paddle_1130, x2paddle_1137, x2paddle_1144, x2paddle_1151,
             x2paddle_1158, x2paddle_1165, x2paddle_1172, x2paddle_1179, x2paddle_1186, x2paddle_1193, x2paddle_1200,
             x2paddle_1207, x2paddle_1214, x2paddle_1221, x2paddle_1228, x2paddle_1235, x2paddle_1242, x2paddle_1249,
             x2paddle_1256, x2paddle_1263, x2paddle_1270, x2paddle_1277, x2paddle_1284, x2paddle_1291], axis=1)
        x2paddle_1293 = fluid.layers.batch_norm(x2paddle_1292, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer27_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer27_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer27_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer27_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1293')
        x2paddle_1294 = fluid.layers.relu(x2paddle_1293, name='x2paddle_1294')
        x2paddle_1295 = fluid.layers.conv2d(x2paddle_1294, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer27_conv1_weight',
                                            name='x2paddle_1295', bias_attr=False)
        x2paddle_1296 = fluid.layers.batch_norm(x2paddle_1295, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer27_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer27_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer27_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer27_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1296')
        x2paddle_1297 = fluid.layers.relu(x2paddle_1296, name='x2paddle_1297')
        x2paddle_1298 = fluid.layers.conv2d(x2paddle_1297, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer27_conv2_weight',
                                            name='x2paddle_1298', bias_attr=False)
        x2paddle_1299 = fluid.layers.concat(
            [x2paddle_1109, x2paddle_1116, x2paddle_1123, x2paddle_1130, x2paddle_1137, x2paddle_1144, x2paddle_1151,
             x2paddle_1158, x2paddle_1165, x2paddle_1172, x2paddle_1179, x2paddle_1186, x2paddle_1193, x2paddle_1200,
             x2paddle_1207, x2paddle_1214, x2paddle_1221, x2paddle_1228, x2paddle_1235, x2paddle_1242, x2paddle_1249,
             x2paddle_1256, x2paddle_1263, x2paddle_1270, x2paddle_1277, x2paddle_1284, x2paddle_1291, x2paddle_1298],
            axis=1)
        x2paddle_1300 = fluid.layers.batch_norm(x2paddle_1299, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer28_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer28_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer28_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer28_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1300')
        x2paddle_1301 = fluid.layers.relu(x2paddle_1300, name='x2paddle_1301')
        x2paddle_1302 = fluid.layers.conv2d(x2paddle_1301, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer28_conv1_weight',
                                            name='x2paddle_1302', bias_attr=False)
        x2paddle_1303 = fluid.layers.batch_norm(x2paddle_1302, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer28_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer28_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer28_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer28_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1303')
        x2paddle_1304 = fluid.layers.relu(x2paddle_1303, name='x2paddle_1304')
        x2paddle_1305 = fluid.layers.conv2d(x2paddle_1304, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer28_conv2_weight',
                                            name='x2paddle_1305', bias_attr=False)
        x2paddle_1306 = fluid.layers.concat(
            [x2paddle_1109, x2paddle_1116, x2paddle_1123, x2paddle_1130, x2paddle_1137, x2paddle_1144, x2paddle_1151,
             x2paddle_1158, x2paddle_1165, x2paddle_1172, x2paddle_1179, x2paddle_1186, x2paddle_1193, x2paddle_1200,
             x2paddle_1207, x2paddle_1214, x2paddle_1221, x2paddle_1228, x2paddle_1235, x2paddle_1242, x2paddle_1249,
             x2paddle_1256, x2paddle_1263, x2paddle_1270, x2paddle_1277, x2paddle_1284, x2paddle_1291, x2paddle_1298,
             x2paddle_1305], axis=1)
        x2paddle_1307 = fluid.layers.batch_norm(x2paddle_1306, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer29_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer29_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer29_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer29_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1307')
        x2paddle_1308 = fluid.layers.relu(x2paddle_1307, name='x2paddle_1308')
        x2paddle_1309 = fluid.layers.conv2d(x2paddle_1308, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer29_conv1_weight',
                                            name='x2paddle_1309', bias_attr=False)
        x2paddle_1310 = fluid.layers.batch_norm(x2paddle_1309, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer29_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer29_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer29_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer29_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1310')
        x2paddle_1311 = fluid.layers.relu(x2paddle_1310, name='x2paddle_1311')
        x2paddle_1312 = fluid.layers.conv2d(x2paddle_1311, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer29_conv2_weight',
                                            name='x2paddle_1312', bias_attr=False)
        x2paddle_1313 = fluid.layers.concat(
            [x2paddle_1109, x2paddle_1116, x2paddle_1123, x2paddle_1130, x2paddle_1137, x2paddle_1144, x2paddle_1151,
             x2paddle_1158, x2paddle_1165, x2paddle_1172, x2paddle_1179, x2paddle_1186, x2paddle_1193, x2paddle_1200,
             x2paddle_1207, x2paddle_1214, x2paddle_1221, x2paddle_1228, x2paddle_1235, x2paddle_1242, x2paddle_1249,
             x2paddle_1256, x2paddle_1263, x2paddle_1270, x2paddle_1277, x2paddle_1284, x2paddle_1291, x2paddle_1298,
             x2paddle_1305, x2paddle_1312], axis=1)
        x2paddle_1314 = fluid.layers.batch_norm(x2paddle_1313, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer30_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer30_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer30_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer30_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1314')
        x2paddle_1315 = fluid.layers.relu(x2paddle_1314, name='x2paddle_1315')
        x2paddle_1316 = fluid.layers.conv2d(x2paddle_1315, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer30_conv1_weight',
                                            name='x2paddle_1316', bias_attr=False)
        x2paddle_1317 = fluid.layers.batch_norm(x2paddle_1316, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer30_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer30_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer30_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer30_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1317')
        x2paddle_1318 = fluid.layers.relu(x2paddle_1317, name='x2paddle_1318')
        x2paddle_1319 = fluid.layers.conv2d(x2paddle_1318, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer30_conv2_weight',
                                            name='x2paddle_1319', bias_attr=False)
        x2paddle_1320 = fluid.layers.concat(
            [x2paddle_1109, x2paddle_1116, x2paddle_1123, x2paddle_1130, x2paddle_1137, x2paddle_1144, x2paddle_1151,
             x2paddle_1158, x2paddle_1165, x2paddle_1172, x2paddle_1179, x2paddle_1186, x2paddle_1193, x2paddle_1200,
             x2paddle_1207, x2paddle_1214, x2paddle_1221, x2paddle_1228, x2paddle_1235, x2paddle_1242, x2paddle_1249,
             x2paddle_1256, x2paddle_1263, x2paddle_1270, x2paddle_1277, x2paddle_1284, x2paddle_1291, x2paddle_1298,
             x2paddle_1305, x2paddle_1312, x2paddle_1319], axis=1)
        x2paddle_1321 = fluid.layers.batch_norm(x2paddle_1320, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer31_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer31_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer31_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer31_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1321')
        x2paddle_1322 = fluid.layers.relu(x2paddle_1321, name='x2paddle_1322')
        x2paddle_1323 = fluid.layers.conv2d(x2paddle_1322, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer31_conv1_weight',
                                            name='x2paddle_1323', bias_attr=False)
        x2paddle_1324 = fluid.layers.batch_norm(x2paddle_1323, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer31_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer31_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer31_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer31_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1324')
        x2paddle_1325 = fluid.layers.relu(x2paddle_1324, name='x2paddle_1325')
        x2paddle_1326 = fluid.layers.conv2d(x2paddle_1325, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer31_conv2_weight',
                                            name='x2paddle_1326', bias_attr=False)
        x2paddle_1327 = fluid.layers.concat(
            [x2paddle_1109, x2paddle_1116, x2paddle_1123, x2paddle_1130, x2paddle_1137, x2paddle_1144, x2paddle_1151,
             x2paddle_1158, x2paddle_1165, x2paddle_1172, x2paddle_1179, x2paddle_1186, x2paddle_1193, x2paddle_1200,
             x2paddle_1207, x2paddle_1214, x2paddle_1221, x2paddle_1228, x2paddle_1235, x2paddle_1242, x2paddle_1249,
             x2paddle_1256, x2paddle_1263, x2paddle_1270, x2paddle_1277, x2paddle_1284, x2paddle_1291, x2paddle_1298,
             x2paddle_1305, x2paddle_1312, x2paddle_1319, x2paddle_1326], axis=1)
        x2paddle_1328 = fluid.layers.batch_norm(x2paddle_1327, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer32_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer32_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer32_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer32_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1328')
        x2paddle_1329 = fluid.layers.relu(x2paddle_1328, name='x2paddle_1329')
        x2paddle_1330 = fluid.layers.conv2d(x2paddle_1329, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer32_conv1_weight',
                                            name='x2paddle_1330', bias_attr=False)
        x2paddle_1331 = fluid.layers.batch_norm(x2paddle_1330, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer32_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer32_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer32_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer32_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1331')
        x2paddle_1332 = fluid.layers.relu(x2paddle_1331, name='x2paddle_1332')
        x2paddle_1333 = fluid.layers.conv2d(x2paddle_1332, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer32_conv2_weight',
                                            name='x2paddle_1333', bias_attr=False)
        x2paddle_1334 = fluid.layers.concat(
            [x2paddle_1109, x2paddle_1116, x2paddle_1123, x2paddle_1130, x2paddle_1137, x2paddle_1144, x2paddle_1151,
             x2paddle_1158, x2paddle_1165, x2paddle_1172, x2paddle_1179, x2paddle_1186, x2paddle_1193, x2paddle_1200,
             x2paddle_1207, x2paddle_1214, x2paddle_1221, x2paddle_1228, x2paddle_1235, x2paddle_1242, x2paddle_1249,
             x2paddle_1256, x2paddle_1263, x2paddle_1270, x2paddle_1277, x2paddle_1284, x2paddle_1291, x2paddle_1298,
             x2paddle_1305, x2paddle_1312, x2paddle_1319, x2paddle_1326, x2paddle_1333], axis=1)
        x2paddle_1335 = fluid.layers.batch_norm(x2paddle_1334, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer33_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer33_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer33_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer33_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1335')
        x2paddle_1336 = fluid.layers.relu(x2paddle_1335, name='x2paddle_1336')
        x2paddle_1337 = fluid.layers.conv2d(x2paddle_1336, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer33_conv1_weight',
                                            name='x2paddle_1337', bias_attr=False)
        x2paddle_1338 = fluid.layers.batch_norm(x2paddle_1337, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer33_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer33_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer33_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer33_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1338')
        x2paddle_1339 = fluid.layers.relu(x2paddle_1338, name='x2paddle_1339')
        x2paddle_1340 = fluid.layers.conv2d(x2paddle_1339, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer33_conv2_weight',
                                            name='x2paddle_1340', bias_attr=False)
        x2paddle_1341 = fluid.layers.concat(
            [x2paddle_1109, x2paddle_1116, x2paddle_1123, x2paddle_1130, x2paddle_1137, x2paddle_1144, x2paddle_1151,
             x2paddle_1158, x2paddle_1165, x2paddle_1172, x2paddle_1179, x2paddle_1186, x2paddle_1193, x2paddle_1200,
             x2paddle_1207, x2paddle_1214, x2paddle_1221, x2paddle_1228, x2paddle_1235, x2paddle_1242, x2paddle_1249,
             x2paddle_1256, x2paddle_1263, x2paddle_1270, x2paddle_1277, x2paddle_1284, x2paddle_1291, x2paddle_1298,
             x2paddle_1305, x2paddle_1312, x2paddle_1319, x2paddle_1326, x2paddle_1333, x2paddle_1340], axis=1)
        x2paddle_1342 = fluid.layers.batch_norm(x2paddle_1341, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer34_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer34_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer34_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer34_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1342')
        x2paddle_1343 = fluid.layers.relu(x2paddle_1342, name='x2paddle_1343')
        x2paddle_1344 = fluid.layers.conv2d(x2paddle_1343, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer34_conv1_weight',
                                            name='x2paddle_1344', bias_attr=False)
        x2paddle_1345 = fluid.layers.batch_norm(x2paddle_1344, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer34_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer34_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer34_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer34_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1345')
        x2paddle_1346 = fluid.layers.relu(x2paddle_1345, name='x2paddle_1346')
        x2paddle_1347 = fluid.layers.conv2d(x2paddle_1346, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer34_conv2_weight',
                                            name='x2paddle_1347', bias_attr=False)
        x2paddle_1348 = fluid.layers.concat(
            [x2paddle_1109, x2paddle_1116, x2paddle_1123, x2paddle_1130, x2paddle_1137, x2paddle_1144, x2paddle_1151,
             x2paddle_1158, x2paddle_1165, x2paddle_1172, x2paddle_1179, x2paddle_1186, x2paddle_1193, x2paddle_1200,
             x2paddle_1207, x2paddle_1214, x2paddle_1221, x2paddle_1228, x2paddle_1235, x2paddle_1242, x2paddle_1249,
             x2paddle_1256, x2paddle_1263, x2paddle_1270, x2paddle_1277, x2paddle_1284, x2paddle_1291, x2paddle_1298,
             x2paddle_1305, x2paddle_1312, x2paddle_1319, x2paddle_1326, x2paddle_1333, x2paddle_1340, x2paddle_1347],
            axis=1)
        x2paddle_1349 = fluid.layers.batch_norm(x2paddle_1348, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer35_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer35_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer35_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer35_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1349')
        x2paddle_1350 = fluid.layers.relu(x2paddle_1349, name='x2paddle_1350')
        x2paddle_1351 = fluid.layers.conv2d(x2paddle_1350, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer35_conv1_weight',
                                            name='x2paddle_1351', bias_attr=False)
        x2paddle_1352 = fluid.layers.batch_norm(x2paddle_1351, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer35_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer35_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer35_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer35_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1352')
        x2paddle_1353 = fluid.layers.relu(x2paddle_1352, name='x2paddle_1353')
        x2paddle_1354 = fluid.layers.conv2d(x2paddle_1353, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer35_conv2_weight',
                                            name='x2paddle_1354', bias_attr=False)
        x2paddle_1355 = fluid.layers.concat(
            [x2paddle_1109, x2paddle_1116, x2paddle_1123, x2paddle_1130, x2paddle_1137, x2paddle_1144, x2paddle_1151,
             x2paddle_1158, x2paddle_1165, x2paddle_1172, x2paddle_1179, x2paddle_1186, x2paddle_1193, x2paddle_1200,
             x2paddle_1207, x2paddle_1214, x2paddle_1221, x2paddle_1228, x2paddle_1235, x2paddle_1242, x2paddle_1249,
             x2paddle_1256, x2paddle_1263, x2paddle_1270, x2paddle_1277, x2paddle_1284, x2paddle_1291, x2paddle_1298,
             x2paddle_1305, x2paddle_1312, x2paddle_1319, x2paddle_1326, x2paddle_1333, x2paddle_1340, x2paddle_1347,
             x2paddle_1354], axis=1)
        x2paddle_1356 = fluid.layers.batch_norm(x2paddle_1355, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer36_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer36_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer36_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer36_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1356')
        x2paddle_1357 = fluid.layers.relu(x2paddle_1356, name='x2paddle_1357')
        x2paddle_1358 = fluid.layers.conv2d(x2paddle_1357, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer36_conv1_weight',
                                            name='x2paddle_1358', bias_attr=False)
        x2paddle_1359 = fluid.layers.batch_norm(x2paddle_1358, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer36_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock3_denselayer36_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock3_denselayer36_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock3_denselayer36_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1359')
        x2paddle_1360 = fluid.layers.relu(x2paddle_1359, name='x2paddle_1360')
        x2paddle_1361 = fluid.layers.conv2d(x2paddle_1360, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock3_denselayer36_conv2_weight',
                                            name='x2paddle_1361', bias_attr=False)
        x2paddle_1362 = fluid.layers.concat(
            [x2paddle_1109, x2paddle_1116, x2paddle_1123, x2paddle_1130, x2paddle_1137, x2paddle_1144, x2paddle_1151,
             x2paddle_1158, x2paddle_1165, x2paddle_1172, x2paddle_1179, x2paddle_1186, x2paddle_1193, x2paddle_1200,
             x2paddle_1207, x2paddle_1214, x2paddle_1221, x2paddle_1228, x2paddle_1235, x2paddle_1242, x2paddle_1249,
             x2paddle_1256, x2paddle_1263, x2paddle_1270, x2paddle_1277, x2paddle_1284, x2paddle_1291, x2paddle_1298,
             x2paddle_1305, x2paddle_1312, x2paddle_1319, x2paddle_1326, x2paddle_1333, x2paddle_1340, x2paddle_1347,
             x2paddle_1354, x2paddle_1361], axis=1)
        x2paddle_1363 = fluid.layers.batch_norm(x2paddle_1362, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_transition3_norm_weight',
                                                bias_attr='x2paddle_fddensenet161_features_transition3_norm_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_transition3_norm_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_transition3_norm_running_var',
                                                use_global_stats=False, name='x2paddle_1363')
        x2paddle_1364 = fluid.layers.relu(x2paddle_1363, name='x2paddle_1364')
        x2paddle_1365 = fluid.layers.conv2d(x2paddle_1364, num_filters=1056, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_transition3_conv_weight',
                                            name='x2paddle_1365', bias_attr=False)
        x2paddle_1366 = fluid.layers.pad2d(x2paddle_1365, pad_value=0.0, mode='constant', paddings=[0, 0, 0, 0],
                                           name='x2paddle_1366')
        x2paddle_1367 = fluid.layers.pool2d(x2paddle_1366, pool_size=[2, 2], pool_type='avg', pool_stride=[2, 2],
                                            pool_padding=[0, 0], ceil_mode=False, exclusive=True, name='x2paddle_1367')
        x2paddle_1368 = fluid.layers.concat([x2paddle_1367], axis=1)
        x2paddle_1369 = fluid.layers.batch_norm(x2paddle_1368, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer1_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer1_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer1_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer1_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1369')
        x2paddle_1370 = fluid.layers.relu(x2paddle_1369, name='x2paddle_1370')
        x2paddle_1371 = fluid.layers.conv2d(x2paddle_1370, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer1_conv1_weight',
                                            name='x2paddle_1371', bias_attr=False)
        x2paddle_1372 = fluid.layers.batch_norm(x2paddle_1371, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer1_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer1_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer1_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer1_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1372')
        x2paddle_1373 = fluid.layers.relu(x2paddle_1372, name='x2paddle_1373')
        x2paddle_1374 = fluid.layers.conv2d(x2paddle_1373, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer1_conv2_weight',
                                            name='x2paddle_1374', bias_attr=False)
        x2paddle_1375 = fluid.layers.concat([x2paddle_1367, x2paddle_1374], axis=1)
        x2paddle_1376 = fluid.layers.batch_norm(x2paddle_1375, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer2_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer2_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer2_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer2_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1376')
        x2paddle_1377 = fluid.layers.relu(x2paddle_1376, name='x2paddle_1377')
        x2paddle_1378 = fluid.layers.conv2d(x2paddle_1377, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer2_conv1_weight',
                                            name='x2paddle_1378', bias_attr=False)
        x2paddle_1379 = fluid.layers.batch_norm(x2paddle_1378, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer2_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer2_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer2_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer2_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1379')
        x2paddle_1380 = fluid.layers.relu(x2paddle_1379, name='x2paddle_1380')
        x2paddle_1381 = fluid.layers.conv2d(x2paddle_1380, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer2_conv2_weight',
                                            name='x2paddle_1381', bias_attr=False)
        x2paddle_1382 = fluid.layers.concat([x2paddle_1367, x2paddle_1374, x2paddle_1381], axis=1)
        x2paddle_1383 = fluid.layers.batch_norm(x2paddle_1382, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer3_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer3_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer3_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer3_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1383')
        x2paddle_1384 = fluid.layers.relu(x2paddle_1383, name='x2paddle_1384')
        x2paddle_1385 = fluid.layers.conv2d(x2paddle_1384, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer3_conv1_weight',
                                            name='x2paddle_1385', bias_attr=False)
        x2paddle_1386 = fluid.layers.batch_norm(x2paddle_1385, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer3_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer3_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer3_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer3_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1386')
        x2paddle_1387 = fluid.layers.relu(x2paddle_1386, name='x2paddle_1387')
        x2paddle_1388 = fluid.layers.conv2d(x2paddle_1387, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer3_conv2_weight',
                                            name='x2paddle_1388', bias_attr=False)
        x2paddle_1389 = fluid.layers.concat([x2paddle_1367, x2paddle_1374, x2paddle_1381, x2paddle_1388], axis=1)
        x2paddle_1390 = fluid.layers.batch_norm(x2paddle_1389, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer4_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer4_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer4_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer4_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1390')
        x2paddle_1391 = fluid.layers.relu(x2paddle_1390, name='x2paddle_1391')
        x2paddle_1392 = fluid.layers.conv2d(x2paddle_1391, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer4_conv1_weight',
                                            name='x2paddle_1392', bias_attr=False)
        x2paddle_1393 = fluid.layers.batch_norm(x2paddle_1392, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer4_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer4_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer4_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer4_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1393')
        x2paddle_1394 = fluid.layers.relu(x2paddle_1393, name='x2paddle_1394')
        x2paddle_1395 = fluid.layers.conv2d(x2paddle_1394, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer4_conv2_weight',
                                            name='x2paddle_1395', bias_attr=False)
        x2paddle_1396 = fluid.layers.concat([x2paddle_1367, x2paddle_1374, x2paddle_1381, x2paddle_1388, x2paddle_1395],
                                            axis=1)
        x2paddle_1397 = fluid.layers.batch_norm(x2paddle_1396, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer5_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer5_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer5_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer5_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1397')
        x2paddle_1398 = fluid.layers.relu(x2paddle_1397, name='x2paddle_1398')
        x2paddle_1399 = fluid.layers.conv2d(x2paddle_1398, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer5_conv1_weight',
                                            name='x2paddle_1399', bias_attr=False)
        x2paddle_1400 = fluid.layers.batch_norm(x2paddle_1399, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer5_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer5_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer5_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer5_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1400')
        x2paddle_1401 = fluid.layers.relu(x2paddle_1400, name='x2paddle_1401')
        x2paddle_1402 = fluid.layers.conv2d(x2paddle_1401, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer5_conv2_weight',
                                            name='x2paddle_1402', bias_attr=False)
        x2paddle_1403 = fluid.layers.concat(
            [x2paddle_1367, x2paddle_1374, x2paddle_1381, x2paddle_1388, x2paddle_1395, x2paddle_1402], axis=1)
        x2paddle_1404 = fluid.layers.batch_norm(x2paddle_1403, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer6_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer6_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer6_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer6_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1404')
        x2paddle_1405 = fluid.layers.relu(x2paddle_1404, name='x2paddle_1405')
        x2paddle_1406 = fluid.layers.conv2d(x2paddle_1405, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer6_conv1_weight',
                                            name='x2paddle_1406', bias_attr=False)
        x2paddle_1407 = fluid.layers.batch_norm(x2paddle_1406, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer6_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer6_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer6_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer6_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1407')
        x2paddle_1408 = fluid.layers.relu(x2paddle_1407, name='x2paddle_1408')
        x2paddle_1409 = fluid.layers.conv2d(x2paddle_1408, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer6_conv2_weight',
                                            name='x2paddle_1409', bias_attr=False)
        x2paddle_1410 = fluid.layers.concat(
            [x2paddle_1367, x2paddle_1374, x2paddle_1381, x2paddle_1388, x2paddle_1395, x2paddle_1402, x2paddle_1409],
            axis=1)
        x2paddle_1411 = fluid.layers.batch_norm(x2paddle_1410, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer7_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer7_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer7_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer7_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1411')
        x2paddle_1412 = fluid.layers.relu(x2paddle_1411, name='x2paddle_1412')
        x2paddle_1413 = fluid.layers.conv2d(x2paddle_1412, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer7_conv1_weight',
                                            name='x2paddle_1413', bias_attr=False)
        x2paddle_1414 = fluid.layers.batch_norm(x2paddle_1413, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer7_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer7_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer7_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer7_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1414')
        x2paddle_1415 = fluid.layers.relu(x2paddle_1414, name='x2paddle_1415')
        x2paddle_1416 = fluid.layers.conv2d(x2paddle_1415, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer7_conv2_weight',
                                            name='x2paddle_1416', bias_attr=False)
        x2paddle_1417 = fluid.layers.concat(
            [x2paddle_1367, x2paddle_1374, x2paddle_1381, x2paddle_1388, x2paddle_1395, x2paddle_1402, x2paddle_1409,
             x2paddle_1416], axis=1)
        x2paddle_1418 = fluid.layers.batch_norm(x2paddle_1417, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer8_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer8_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer8_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer8_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1418')
        x2paddle_1419 = fluid.layers.relu(x2paddle_1418, name='x2paddle_1419')
        x2paddle_1420 = fluid.layers.conv2d(x2paddle_1419, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer8_conv1_weight',
                                            name='x2paddle_1420', bias_attr=False)
        x2paddle_1421 = fluid.layers.batch_norm(x2paddle_1420, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer8_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer8_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer8_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer8_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1421')
        x2paddle_1422 = fluid.layers.relu(x2paddle_1421, name='x2paddle_1422')
        x2paddle_1423 = fluid.layers.conv2d(x2paddle_1422, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer8_conv2_weight',
                                            name='x2paddle_1423', bias_attr=False)
        x2paddle_1424 = fluid.layers.concat(
            [x2paddle_1367, x2paddle_1374, x2paddle_1381, x2paddle_1388, x2paddle_1395, x2paddle_1402, x2paddle_1409,
             x2paddle_1416, x2paddle_1423], axis=1)
        x2paddle_1425 = fluid.layers.batch_norm(x2paddle_1424, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer9_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer9_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer9_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer9_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1425')
        x2paddle_1426 = fluid.layers.relu(x2paddle_1425, name='x2paddle_1426')
        x2paddle_1427 = fluid.layers.conv2d(x2paddle_1426, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer9_conv1_weight',
                                            name='x2paddle_1427', bias_attr=False)
        x2paddle_1428 = fluid.layers.batch_norm(x2paddle_1427, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer9_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer9_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer9_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer9_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1428')
        x2paddle_1429 = fluid.layers.relu(x2paddle_1428, name='x2paddle_1429')
        x2paddle_1430 = fluid.layers.conv2d(x2paddle_1429, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer9_conv2_weight',
                                            name='x2paddle_1430', bias_attr=False)
        x2paddle_1431 = fluid.layers.concat(
            [x2paddle_1367, x2paddle_1374, x2paddle_1381, x2paddle_1388, x2paddle_1395, x2paddle_1402, x2paddle_1409,
             x2paddle_1416, x2paddle_1423, x2paddle_1430], axis=1)
        x2paddle_1432 = fluid.layers.batch_norm(x2paddle_1431, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer10_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer10_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer10_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer10_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1432')
        x2paddle_1433 = fluid.layers.relu(x2paddle_1432, name='x2paddle_1433')
        x2paddle_1434 = fluid.layers.conv2d(x2paddle_1433, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer10_conv1_weight',
                                            name='x2paddle_1434', bias_attr=False)
        x2paddle_1435 = fluid.layers.batch_norm(x2paddle_1434, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer10_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer10_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer10_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer10_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1435')
        x2paddle_1436 = fluid.layers.relu(x2paddle_1435, name='x2paddle_1436')
        x2paddle_1437 = fluid.layers.conv2d(x2paddle_1436, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer10_conv2_weight',
                                            name='x2paddle_1437', bias_attr=False)
        x2paddle_1438 = fluid.layers.concat(
            [x2paddle_1367, x2paddle_1374, x2paddle_1381, x2paddle_1388, x2paddle_1395, x2paddle_1402, x2paddle_1409,
             x2paddle_1416, x2paddle_1423, x2paddle_1430, x2paddle_1437], axis=1)
        x2paddle_1439 = fluid.layers.batch_norm(x2paddle_1438, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer11_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer11_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer11_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer11_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1439')
        x2paddle_1440 = fluid.layers.relu(x2paddle_1439, name='x2paddle_1440')
        x2paddle_1441 = fluid.layers.conv2d(x2paddle_1440, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer11_conv1_weight',
                                            name='x2paddle_1441', bias_attr=False)
        x2paddle_1442 = fluid.layers.batch_norm(x2paddle_1441, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer11_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer11_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer11_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer11_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1442')
        x2paddle_1443 = fluid.layers.relu(x2paddle_1442, name='x2paddle_1443')
        x2paddle_1444 = fluid.layers.conv2d(x2paddle_1443, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer11_conv2_weight',
                                            name='x2paddle_1444', bias_attr=False)
        x2paddle_1445 = fluid.layers.concat(
            [x2paddle_1367, x2paddle_1374, x2paddle_1381, x2paddle_1388, x2paddle_1395, x2paddle_1402, x2paddle_1409,
             x2paddle_1416, x2paddle_1423, x2paddle_1430, x2paddle_1437, x2paddle_1444], axis=1)
        x2paddle_1446 = fluid.layers.batch_norm(x2paddle_1445, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer12_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer12_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer12_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer12_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1446')
        x2paddle_1447 = fluid.layers.relu(x2paddle_1446, name='x2paddle_1447')
        x2paddle_1448 = fluid.layers.conv2d(x2paddle_1447, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer12_conv1_weight',
                                            name='x2paddle_1448', bias_attr=False)
        x2paddle_1449 = fluid.layers.batch_norm(x2paddle_1448, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer12_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer12_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer12_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer12_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1449')
        x2paddle_1450 = fluid.layers.relu(x2paddle_1449, name='x2paddle_1450')
        x2paddle_1451 = fluid.layers.conv2d(x2paddle_1450, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer12_conv2_weight',
                                            name='x2paddle_1451', bias_attr=False)
        x2paddle_1452 = fluid.layers.concat(
            [x2paddle_1367, x2paddle_1374, x2paddle_1381, x2paddle_1388, x2paddle_1395, x2paddle_1402, x2paddle_1409,
             x2paddle_1416, x2paddle_1423, x2paddle_1430, x2paddle_1437, x2paddle_1444, x2paddle_1451], axis=1)
        x2paddle_1453 = fluid.layers.batch_norm(x2paddle_1452, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer13_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer13_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer13_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer13_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1453')
        x2paddle_1454 = fluid.layers.relu(x2paddle_1453, name='x2paddle_1454')
        x2paddle_1455 = fluid.layers.conv2d(x2paddle_1454, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer13_conv1_weight',
                                            name='x2paddle_1455', bias_attr=False)
        x2paddle_1456 = fluid.layers.batch_norm(x2paddle_1455, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer13_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer13_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer13_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer13_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1456')
        x2paddle_1457 = fluid.layers.relu(x2paddle_1456, name='x2paddle_1457')
        x2paddle_1458 = fluid.layers.conv2d(x2paddle_1457, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer13_conv2_weight',
                                            name='x2paddle_1458', bias_attr=False)
        x2paddle_1459 = fluid.layers.concat(
            [x2paddle_1367, x2paddle_1374, x2paddle_1381, x2paddle_1388, x2paddle_1395, x2paddle_1402, x2paddle_1409,
             x2paddle_1416, x2paddle_1423, x2paddle_1430, x2paddle_1437, x2paddle_1444, x2paddle_1451, x2paddle_1458],
            axis=1)
        x2paddle_1460 = fluid.layers.batch_norm(x2paddle_1459, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer14_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer14_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer14_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer14_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1460')
        x2paddle_1461 = fluid.layers.relu(x2paddle_1460, name='x2paddle_1461')
        x2paddle_1462 = fluid.layers.conv2d(x2paddle_1461, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer14_conv1_weight',
                                            name='x2paddle_1462', bias_attr=False)
        x2paddle_1463 = fluid.layers.batch_norm(x2paddle_1462, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer14_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer14_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer14_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer14_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1463')
        x2paddle_1464 = fluid.layers.relu(x2paddle_1463, name='x2paddle_1464')
        x2paddle_1465 = fluid.layers.conv2d(x2paddle_1464, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer14_conv2_weight',
                                            name='x2paddle_1465', bias_attr=False)
        x2paddle_1466 = fluid.layers.concat(
            [x2paddle_1367, x2paddle_1374, x2paddle_1381, x2paddle_1388, x2paddle_1395, x2paddle_1402, x2paddle_1409,
             x2paddle_1416, x2paddle_1423, x2paddle_1430, x2paddle_1437, x2paddle_1444, x2paddle_1451, x2paddle_1458,
             x2paddle_1465], axis=1)
        x2paddle_1467 = fluid.layers.batch_norm(x2paddle_1466, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer15_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer15_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer15_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer15_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1467')
        x2paddle_1468 = fluid.layers.relu(x2paddle_1467, name='x2paddle_1468')
        x2paddle_1469 = fluid.layers.conv2d(x2paddle_1468, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer15_conv1_weight',
                                            name='x2paddle_1469', bias_attr=False)
        x2paddle_1470 = fluid.layers.batch_norm(x2paddle_1469, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer15_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer15_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer15_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer15_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1470')
        x2paddle_1471 = fluid.layers.relu(x2paddle_1470, name='x2paddle_1471')
        x2paddle_1472 = fluid.layers.conv2d(x2paddle_1471, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer15_conv2_weight',
                                            name='x2paddle_1472', bias_attr=False)
        x2paddle_1473 = fluid.layers.concat(
            [x2paddle_1367, x2paddle_1374, x2paddle_1381, x2paddle_1388, x2paddle_1395, x2paddle_1402, x2paddle_1409,
             x2paddle_1416, x2paddle_1423, x2paddle_1430, x2paddle_1437, x2paddle_1444, x2paddle_1451, x2paddle_1458,
             x2paddle_1465, x2paddle_1472], axis=1)
        x2paddle_1474 = fluid.layers.batch_norm(x2paddle_1473, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer16_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer16_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer16_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer16_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1474')
        x2paddle_1475 = fluid.layers.relu(x2paddle_1474, name='x2paddle_1475')
        x2paddle_1476 = fluid.layers.conv2d(x2paddle_1475, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer16_conv1_weight',
                                            name='x2paddle_1476', bias_attr=False)
        x2paddle_1477 = fluid.layers.batch_norm(x2paddle_1476, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer16_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer16_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer16_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer16_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1477')
        x2paddle_1478 = fluid.layers.relu(x2paddle_1477, name='x2paddle_1478')
        x2paddle_1479 = fluid.layers.conv2d(x2paddle_1478, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer16_conv2_weight',
                                            name='x2paddle_1479', bias_attr=False)
        x2paddle_1480 = fluid.layers.concat(
            [x2paddle_1367, x2paddle_1374, x2paddle_1381, x2paddle_1388, x2paddle_1395, x2paddle_1402, x2paddle_1409,
             x2paddle_1416, x2paddle_1423, x2paddle_1430, x2paddle_1437, x2paddle_1444, x2paddle_1451, x2paddle_1458,
             x2paddle_1465, x2paddle_1472, x2paddle_1479], axis=1)
        x2paddle_1481 = fluid.layers.batch_norm(x2paddle_1480, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer17_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer17_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer17_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer17_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1481')
        x2paddle_1482 = fluid.layers.relu(x2paddle_1481, name='x2paddle_1482')
        x2paddle_1483 = fluid.layers.conv2d(x2paddle_1482, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer17_conv1_weight',
                                            name='x2paddle_1483', bias_attr=False)
        x2paddle_1484 = fluid.layers.batch_norm(x2paddle_1483, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer17_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer17_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer17_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer17_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1484')
        x2paddle_1485 = fluid.layers.relu(x2paddle_1484, name='x2paddle_1485')
        x2paddle_1486 = fluid.layers.conv2d(x2paddle_1485, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer17_conv2_weight',
                                            name='x2paddle_1486', bias_attr=False)
        x2paddle_1487 = fluid.layers.concat(
            [x2paddle_1367, x2paddle_1374, x2paddle_1381, x2paddle_1388, x2paddle_1395, x2paddle_1402, x2paddle_1409,
             x2paddle_1416, x2paddle_1423, x2paddle_1430, x2paddle_1437, x2paddle_1444, x2paddle_1451, x2paddle_1458,
             x2paddle_1465, x2paddle_1472, x2paddle_1479, x2paddle_1486], axis=1)
        x2paddle_1488 = fluid.layers.batch_norm(x2paddle_1487, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer18_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer18_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer18_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer18_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1488')
        x2paddle_1489 = fluid.layers.relu(x2paddle_1488, name='x2paddle_1489')
        x2paddle_1490 = fluid.layers.conv2d(x2paddle_1489, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer18_conv1_weight',
                                            name='x2paddle_1490', bias_attr=False)
        x2paddle_1491 = fluid.layers.batch_norm(x2paddle_1490, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer18_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer18_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer18_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer18_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1491')
        x2paddle_1492 = fluid.layers.relu(x2paddle_1491, name='x2paddle_1492')
        x2paddle_1493 = fluid.layers.conv2d(x2paddle_1492, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer18_conv2_weight',
                                            name='x2paddle_1493', bias_attr=False)
        x2paddle_1494 = fluid.layers.concat(
            [x2paddle_1367, x2paddle_1374, x2paddle_1381, x2paddle_1388, x2paddle_1395, x2paddle_1402, x2paddle_1409,
             x2paddle_1416, x2paddle_1423, x2paddle_1430, x2paddle_1437, x2paddle_1444, x2paddle_1451, x2paddle_1458,
             x2paddle_1465, x2paddle_1472, x2paddle_1479, x2paddle_1486, x2paddle_1493], axis=1)
        x2paddle_1495 = fluid.layers.batch_norm(x2paddle_1494, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer19_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer19_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer19_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer19_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1495')
        x2paddle_1496 = fluid.layers.relu(x2paddle_1495, name='x2paddle_1496')
        x2paddle_1497 = fluid.layers.conv2d(x2paddle_1496, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer19_conv1_weight',
                                            name='x2paddle_1497', bias_attr=False)
        x2paddle_1498 = fluid.layers.batch_norm(x2paddle_1497, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer19_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer19_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer19_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer19_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1498')
        x2paddle_1499 = fluid.layers.relu(x2paddle_1498, name='x2paddle_1499')
        x2paddle_1500 = fluid.layers.conv2d(x2paddle_1499, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer19_conv2_weight',
                                            name='x2paddle_1500', bias_attr=False)
        x2paddle_1501 = fluid.layers.concat(
            [x2paddle_1367, x2paddle_1374, x2paddle_1381, x2paddle_1388, x2paddle_1395, x2paddle_1402, x2paddle_1409,
             x2paddle_1416, x2paddle_1423, x2paddle_1430, x2paddle_1437, x2paddle_1444, x2paddle_1451, x2paddle_1458,
             x2paddle_1465, x2paddle_1472, x2paddle_1479, x2paddle_1486, x2paddle_1493, x2paddle_1500], axis=1)
        x2paddle_1502 = fluid.layers.batch_norm(x2paddle_1501, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer20_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer20_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer20_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer20_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1502')
        x2paddle_1503 = fluid.layers.relu(x2paddle_1502, name='x2paddle_1503')
        x2paddle_1504 = fluid.layers.conv2d(x2paddle_1503, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer20_conv1_weight',
                                            name='x2paddle_1504', bias_attr=False)
        x2paddle_1505 = fluid.layers.batch_norm(x2paddle_1504, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer20_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer20_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer20_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer20_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1505')
        x2paddle_1506 = fluid.layers.relu(x2paddle_1505, name='x2paddle_1506')
        x2paddle_1507 = fluid.layers.conv2d(x2paddle_1506, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer20_conv2_weight',
                                            name='x2paddle_1507', bias_attr=False)
        x2paddle_1508 = fluid.layers.concat(
            [x2paddle_1367, x2paddle_1374, x2paddle_1381, x2paddle_1388, x2paddle_1395, x2paddle_1402, x2paddle_1409,
             x2paddle_1416, x2paddle_1423, x2paddle_1430, x2paddle_1437, x2paddle_1444, x2paddle_1451, x2paddle_1458,
             x2paddle_1465, x2paddle_1472, x2paddle_1479, x2paddle_1486, x2paddle_1493, x2paddle_1500, x2paddle_1507],
            axis=1)
        x2paddle_1509 = fluid.layers.batch_norm(x2paddle_1508, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer21_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer21_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer21_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer21_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1509')
        x2paddle_1510 = fluid.layers.relu(x2paddle_1509, name='x2paddle_1510')
        x2paddle_1511 = fluid.layers.conv2d(x2paddle_1510, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer21_conv1_weight',
                                            name='x2paddle_1511', bias_attr=False)
        x2paddle_1512 = fluid.layers.batch_norm(x2paddle_1511, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer21_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer21_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer21_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer21_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1512')
        x2paddle_1513 = fluid.layers.relu(x2paddle_1512, name='x2paddle_1513')
        x2paddle_1514 = fluid.layers.conv2d(x2paddle_1513, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer21_conv2_weight',
                                            name='x2paddle_1514', bias_attr=False)
        x2paddle_1515 = fluid.layers.concat(
            [x2paddle_1367, x2paddle_1374, x2paddle_1381, x2paddle_1388, x2paddle_1395, x2paddle_1402, x2paddle_1409,
             x2paddle_1416, x2paddle_1423, x2paddle_1430, x2paddle_1437, x2paddle_1444, x2paddle_1451, x2paddle_1458,
             x2paddle_1465, x2paddle_1472, x2paddle_1479, x2paddle_1486, x2paddle_1493, x2paddle_1500, x2paddle_1507,
             x2paddle_1514], axis=1)
        x2paddle_1516 = fluid.layers.batch_norm(x2paddle_1515, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer22_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer22_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer22_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer22_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1516')
        x2paddle_1517 = fluid.layers.relu(x2paddle_1516, name='x2paddle_1517')
        x2paddle_1518 = fluid.layers.conv2d(x2paddle_1517, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer22_conv1_weight',
                                            name='x2paddle_1518', bias_attr=False)
        x2paddle_1519 = fluid.layers.batch_norm(x2paddle_1518, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer22_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer22_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer22_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer22_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1519')
        x2paddle_1520 = fluid.layers.relu(x2paddle_1519, name='x2paddle_1520')
        x2paddle_1521 = fluid.layers.conv2d(x2paddle_1520, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer22_conv2_weight',
                                            name='x2paddle_1521', bias_attr=False)
        x2paddle_1522 = fluid.layers.concat(
            [x2paddle_1367, x2paddle_1374, x2paddle_1381, x2paddle_1388, x2paddle_1395, x2paddle_1402, x2paddle_1409,
             x2paddle_1416, x2paddle_1423, x2paddle_1430, x2paddle_1437, x2paddle_1444, x2paddle_1451, x2paddle_1458,
             x2paddle_1465, x2paddle_1472, x2paddle_1479, x2paddle_1486, x2paddle_1493, x2paddle_1500, x2paddle_1507,
             x2paddle_1514, x2paddle_1521], axis=1)
        x2paddle_1523 = fluid.layers.batch_norm(x2paddle_1522, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer23_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer23_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer23_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer23_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1523')
        x2paddle_1524 = fluid.layers.relu(x2paddle_1523, name='x2paddle_1524')
        x2paddle_1525 = fluid.layers.conv2d(x2paddle_1524, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer23_conv1_weight',
                                            name='x2paddle_1525', bias_attr=False)
        x2paddle_1526 = fluid.layers.batch_norm(x2paddle_1525, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer23_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer23_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer23_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer23_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1526')
        x2paddle_1527 = fluid.layers.relu(x2paddle_1526, name='x2paddle_1527')
        x2paddle_1528 = fluid.layers.conv2d(x2paddle_1527, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer23_conv2_weight',
                                            name='x2paddle_1528', bias_attr=False)
        x2paddle_1529 = fluid.layers.concat(
            [x2paddle_1367, x2paddle_1374, x2paddle_1381, x2paddle_1388, x2paddle_1395, x2paddle_1402, x2paddle_1409,
             x2paddle_1416, x2paddle_1423, x2paddle_1430, x2paddle_1437, x2paddle_1444, x2paddle_1451, x2paddle_1458,
             x2paddle_1465, x2paddle_1472, x2paddle_1479, x2paddle_1486, x2paddle_1493, x2paddle_1500, x2paddle_1507,
             x2paddle_1514, x2paddle_1521, x2paddle_1528], axis=1)
        x2paddle_1530 = fluid.layers.batch_norm(x2paddle_1529, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer24_norm1_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer24_norm1_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer24_norm1_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer24_norm1_running_var',
                                                use_global_stats=False, name='x2paddle_1530')
        x2paddle_1531 = fluid.layers.relu(x2paddle_1530, name='x2paddle_1531')
        x2paddle_1532 = fluid.layers.conv2d(x2paddle_1531, num_filters=192, filter_size=[1, 1], stride=[1, 1],
                                            padding=[0, 0], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer24_conv1_weight',
                                            name='x2paddle_1532', bias_attr=False)
        x2paddle_1533 = fluid.layers.batch_norm(x2paddle_1532, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer24_norm2_weight',
                                                bias_attr='x2paddle_fddensenet161_features_denseblock4_denselayer24_norm2_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_denseblock4_denselayer24_norm2_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_denseblock4_denselayer24_norm2_running_var',
                                                use_global_stats=False, name='x2paddle_1533')
        x2paddle_1534 = fluid.layers.relu(x2paddle_1533, name='x2paddle_1534')
        x2paddle_1535 = fluid.layers.conv2d(x2paddle_1534, num_filters=48, filter_size=[3, 3], stride=[1, 1],
                                            padding=[1, 1], dilation=[1, 1], groups=1,
                                            param_attr='x2paddle_fddensenet161_features_denseblock4_denselayer24_conv2_weight',
                                            name='x2paddle_1535', bias_attr=False)
        x2paddle_1536 = fluid.layers.concat(
            [x2paddle_1367, x2paddle_1374, x2paddle_1381, x2paddle_1388, x2paddle_1395, x2paddle_1402, x2paddle_1409,
             x2paddle_1416, x2paddle_1423, x2paddle_1430, x2paddle_1437, x2paddle_1444, x2paddle_1451, x2paddle_1458,
             x2paddle_1465, x2paddle_1472, x2paddle_1479, x2paddle_1486, x2paddle_1493, x2paddle_1500, x2paddle_1507,
             x2paddle_1514, x2paddle_1521, x2paddle_1528, x2paddle_1535], axis=1)
        x2paddle_1537 = fluid.layers.batch_norm(x2paddle_1536, momentum=0.8999999761581421,
                                                epsilon=9.999999747378752e-06, data_layout='NCHW', is_test=True,
                                                param_attr='x2paddle_fddensenet161_features_norm5_weight',
                                                bias_attr='x2paddle_fddensenet161_features_norm5_bias',
                                                moving_mean_name='x2paddle_fddensenet161_features_norm5_running_mean',
                                                moving_variance_name='x2paddle_fddensenet161_features_norm5_running_var',
                                                use_global_stats=False, name='x2paddle_1537')
        x2paddle_1538 = fluid.layers.relu(x2paddle_1537, name='x2paddle_1538')
        x2paddle_1539 = fluid.layers.pool2d(x2paddle_1538, pool_type='avg', global_pooling=True, name='x2paddle_1539')
        x2paddle_1540 = fluid.layers.flatten(x2paddle_1539, axis=1, name='x2paddle_1540')
        x2paddle_output_mm = fluid.layers.matmul(x=x2paddle_1540, y=x2paddle_fddensenet161_classifier_weight,
                                                 transpose_x=False, transpose_y=True, alpha=1.0,
                                                 name='x2paddle_output_mm')
        x2paddle_output = fluid.layers.elementwise_add(x=x2paddle_output_mm, y=x2paddle_fddensenet161_classifier_bias,
                                                       name='x2paddle_output')

        return x2paddle_output


def fddensenet():
    return Fddensenet()