from paddle.fluid.initializer import Constant
from paddle.fluid.param_attr import ParamAttr
import paddle.fluid as fluid

class Fdresnext():
    def __init__(self):
        pass

    def net(self, x2paddle_input):
        x2paddle_fdresnext_fc_bias = fluid.layers.create_parameter(dtype='float32', shape=[121],
                                                                   name='x2paddle_fdresnext_fc_bias',
                                                                   attr='x2paddle_fdresnext_fc_bias',
                                                                   default_initializer=Constant(0.0))
        x2paddle_fdresnext_fc_weight = fluid.layers.create_parameter(dtype='float32', shape=[121, 2048],
                                                                     name='x2paddle_fdresnext_fc_weight',
                                                                     attr='x2paddle_fdresnext_fc_weight',
                                                                     default_initializer=Constant(0.0))
        x2paddle_321 = fluid.layers.conv2d(x2paddle_input, num_filters=64, filter_size=[7, 7], stride=[2, 2],
                                           padding=[3, 3], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_conv1_weight', name='x2paddle_321',
                                           bias_attr=False)
        x2paddle_322 = fluid.layers.batch_norm(x2paddle_321, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_bn1_weight',
                                               bias_attr='x2paddle_fdresnext_bn1_bias',
                                               moving_mean_name='x2paddle_fdresnext_bn1_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_bn1_running_var',
                                               use_global_stats=False, name='x2paddle_322')
        x2paddle_323 = fluid.layers.relu(x2paddle_322, name='x2paddle_323')
        x2paddle_324 = fluid.layers.pool2d(x2paddle_323, pool_size=[3, 3], pool_type='max', pool_stride=[2, 2],
                                           pool_padding=[1, 1], ceil_mode=False, name='x2paddle_324', exclusive=False)
        x2paddle_325 = fluid.layers.conv2d(x2paddle_324, num_filters=128, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer1_0_conv1_weight', name='x2paddle_325',
                                           bias_attr=False)
        x2paddle_333 = fluid.layers.conv2d(x2paddle_324, num_filters=256, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer1_0_downsample_0_weight',
                                           name='x2paddle_333', bias_attr=False)
        x2paddle_326 = fluid.layers.batch_norm(x2paddle_325, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer1_0_bn1_weight',
                                               bias_attr='x2paddle_fdresnext_layer1_0_bn1_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer1_0_bn1_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer1_0_bn1_running_var',
                                               use_global_stats=False, name='x2paddle_326')
        x2paddle_334 = fluid.layers.batch_norm(x2paddle_333, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer1_0_downsample_1_weight',
                                               bias_attr='x2paddle_fdresnext_layer1_0_downsample_1_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer1_0_downsample_1_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer1_0_downsample_1_running_var',
                                               use_global_stats=False, name='x2paddle_334')
        x2paddle_327 = fluid.layers.relu(x2paddle_326, name='x2paddle_327')
        x2paddle_328 = fluid.layers.conv2d(x2paddle_327, num_filters=128, filter_size=[3, 3], stride=[1, 1],
                                           padding=[1, 1], dilation=[1, 1], groups=32,
                                           param_attr='x2paddle_fdresnext_layer1_0_conv2_weight', name='x2paddle_328',
                                           bias_attr=False)
        x2paddle_329 = fluid.layers.batch_norm(x2paddle_328, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer1_0_bn2_weight',
                                               bias_attr='x2paddle_fdresnext_layer1_0_bn2_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer1_0_bn2_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer1_0_bn2_running_var',
                                               use_global_stats=False, name='x2paddle_329')
        x2paddle_330 = fluid.layers.relu(x2paddle_329, name='x2paddle_330')
        x2paddle_331 = fluid.layers.conv2d(x2paddle_330, num_filters=256, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer1_0_conv3_weight', name='x2paddle_331',
                                           bias_attr=False)
        x2paddle_332 = fluid.layers.batch_norm(x2paddle_331, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer1_0_bn3_weight',
                                               bias_attr='x2paddle_fdresnext_layer1_0_bn3_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer1_0_bn3_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer1_0_bn3_running_var',
                                               use_global_stats=False, name='x2paddle_332')
        x2paddle_335 = fluid.layers.elementwise_add(x=x2paddle_332, y=x2paddle_334, name='x2paddle_335')
        x2paddle_336 = fluid.layers.relu(x2paddle_335, name='x2paddle_336')
        x2paddle_337 = fluid.layers.conv2d(x2paddle_336, num_filters=128, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer1_1_conv1_weight', name='x2paddle_337',
                                           bias_attr=False)
        x2paddle_338 = fluid.layers.batch_norm(x2paddle_337, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer1_1_bn1_weight',
                                               bias_attr='x2paddle_fdresnext_layer1_1_bn1_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer1_1_bn1_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer1_1_bn1_running_var',
                                               use_global_stats=False, name='x2paddle_338')
        x2paddle_339 = fluid.layers.relu(x2paddle_338, name='x2paddle_339')
        x2paddle_340 = fluid.layers.conv2d(x2paddle_339, num_filters=128, filter_size=[3, 3], stride=[1, 1],
                                           padding=[1, 1], dilation=[1, 1], groups=32,
                                           param_attr='x2paddle_fdresnext_layer1_1_conv2_weight', name='x2paddle_340',
                                           bias_attr=False)
        x2paddle_341 = fluid.layers.batch_norm(x2paddle_340, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer1_1_bn2_weight',
                                               bias_attr='x2paddle_fdresnext_layer1_1_bn2_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer1_1_bn2_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer1_1_bn2_running_var',
                                               use_global_stats=False, name='x2paddle_341')
        x2paddle_342 = fluid.layers.relu(x2paddle_341, name='x2paddle_342')
        x2paddle_343 = fluid.layers.conv2d(x2paddle_342, num_filters=256, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer1_1_conv3_weight', name='x2paddle_343',
                                           bias_attr=False)
        x2paddle_344 = fluid.layers.batch_norm(x2paddle_343, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer1_1_bn3_weight',
                                               bias_attr='x2paddle_fdresnext_layer1_1_bn3_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer1_1_bn3_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer1_1_bn3_running_var',
                                               use_global_stats=False, name='x2paddle_344')
        x2paddle_345 = fluid.layers.elementwise_add(x=x2paddle_344, y=x2paddle_336, name='x2paddle_345')
        x2paddle_346 = fluid.layers.relu(x2paddle_345, name='x2paddle_346')
        x2paddle_347 = fluid.layers.conv2d(x2paddle_346, num_filters=128, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer1_2_conv1_weight', name='x2paddle_347',
                                           bias_attr=False)
        x2paddle_348 = fluid.layers.batch_norm(x2paddle_347, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer1_2_bn1_weight',
                                               bias_attr='x2paddle_fdresnext_layer1_2_bn1_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer1_2_bn1_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer1_2_bn1_running_var',
                                               use_global_stats=False, name='x2paddle_348')
        x2paddle_349 = fluid.layers.relu(x2paddle_348, name='x2paddle_349')
        x2paddle_350 = fluid.layers.conv2d(x2paddle_349, num_filters=128, filter_size=[3, 3], stride=[1, 1],
                                           padding=[1, 1], dilation=[1, 1], groups=32,
                                           param_attr='x2paddle_fdresnext_layer1_2_conv2_weight', name='x2paddle_350',
                                           bias_attr=False)
        x2paddle_351 = fluid.layers.batch_norm(x2paddle_350, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer1_2_bn2_weight',
                                               bias_attr='x2paddle_fdresnext_layer1_2_bn2_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer1_2_bn2_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer1_2_bn2_running_var',
                                               use_global_stats=False, name='x2paddle_351')
        x2paddle_352 = fluid.layers.relu(x2paddle_351, name='x2paddle_352')
        x2paddle_353 = fluid.layers.conv2d(x2paddle_352, num_filters=256, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer1_2_conv3_weight', name='x2paddle_353',
                                           bias_attr=False)
        x2paddle_354 = fluid.layers.batch_norm(x2paddle_353, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer1_2_bn3_weight',
                                               bias_attr='x2paddle_fdresnext_layer1_2_bn3_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer1_2_bn3_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer1_2_bn3_running_var',
                                               use_global_stats=False, name='x2paddle_354')
        x2paddle_355 = fluid.layers.elementwise_add(x=x2paddle_354, y=x2paddle_346, name='x2paddle_355')
        x2paddle_356 = fluid.layers.relu(x2paddle_355, name='x2paddle_356')
        x2paddle_357 = fluid.layers.conv2d(x2paddle_356, num_filters=256, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer2_0_conv1_weight', name='x2paddle_357',
                                           bias_attr=False)
        x2paddle_365 = fluid.layers.conv2d(x2paddle_356, num_filters=512, filter_size=[1, 1], stride=[2, 2],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer2_0_downsample_0_weight',
                                           name='x2paddle_365', bias_attr=False)
        x2paddle_358 = fluid.layers.batch_norm(x2paddle_357, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer2_0_bn1_weight',
                                               bias_attr='x2paddle_fdresnext_layer2_0_bn1_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer2_0_bn1_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer2_0_bn1_running_var',
                                               use_global_stats=False, name='x2paddle_358')
        x2paddle_366 = fluid.layers.batch_norm(x2paddle_365, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer2_0_downsample_1_weight',
                                               bias_attr='x2paddle_fdresnext_layer2_0_downsample_1_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer2_0_downsample_1_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer2_0_downsample_1_running_var',
                                               use_global_stats=False, name='x2paddle_366')
        x2paddle_359 = fluid.layers.relu(x2paddle_358, name='x2paddle_359')
        x2paddle_360 = fluid.layers.conv2d(x2paddle_359, num_filters=256, filter_size=[3, 3], stride=[2, 2],
                                           padding=[1, 1], dilation=[1, 1], groups=32,
                                           param_attr='x2paddle_fdresnext_layer2_0_conv2_weight', name='x2paddle_360',
                                           bias_attr=False)
        x2paddle_361 = fluid.layers.batch_norm(x2paddle_360, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer2_0_bn2_weight',
                                               bias_attr='x2paddle_fdresnext_layer2_0_bn2_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer2_0_bn2_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer2_0_bn2_running_var',
                                               use_global_stats=False, name='x2paddle_361')
        x2paddle_362 = fluid.layers.relu(x2paddle_361, name='x2paddle_362')
        x2paddle_363 = fluid.layers.conv2d(x2paddle_362, num_filters=512, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer2_0_conv3_weight', name='x2paddle_363',
                                           bias_attr=False)
        x2paddle_364 = fluid.layers.batch_norm(x2paddle_363, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer2_0_bn3_weight',
                                               bias_attr='x2paddle_fdresnext_layer2_0_bn3_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer2_0_bn3_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer2_0_bn3_running_var',
                                               use_global_stats=False, name='x2paddle_364')
        x2paddle_367 = fluid.layers.elementwise_add(x=x2paddle_364, y=x2paddle_366, name='x2paddle_367')
        x2paddle_368 = fluid.layers.relu(x2paddle_367, name='x2paddle_368')
        x2paddle_369 = fluid.layers.conv2d(x2paddle_368, num_filters=256, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer2_1_conv1_weight', name='x2paddle_369',
                                           bias_attr=False)
        x2paddle_370 = fluid.layers.batch_norm(x2paddle_369, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer2_1_bn1_weight',
                                               bias_attr='x2paddle_fdresnext_layer2_1_bn1_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer2_1_bn1_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer2_1_bn1_running_var',
                                               use_global_stats=False, name='x2paddle_370')
        x2paddle_371 = fluid.layers.relu(x2paddle_370, name='x2paddle_371')
        x2paddle_372 = fluid.layers.conv2d(x2paddle_371, num_filters=256, filter_size=[3, 3], stride=[1, 1],
                                           padding=[1, 1], dilation=[1, 1], groups=32,
                                           param_attr='x2paddle_fdresnext_layer2_1_conv2_weight', name='x2paddle_372',
                                           bias_attr=False)
        x2paddle_373 = fluid.layers.batch_norm(x2paddle_372, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer2_1_bn2_weight',
                                               bias_attr='x2paddle_fdresnext_layer2_1_bn2_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer2_1_bn2_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer2_1_bn2_running_var',
                                               use_global_stats=False, name='x2paddle_373')
        x2paddle_374 = fluid.layers.relu(x2paddle_373, name='x2paddle_374')
        x2paddle_375 = fluid.layers.conv2d(x2paddle_374, num_filters=512, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer2_1_conv3_weight', name='x2paddle_375',
                                           bias_attr=False)
        x2paddle_376 = fluid.layers.batch_norm(x2paddle_375, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer2_1_bn3_weight',
                                               bias_attr='x2paddle_fdresnext_layer2_1_bn3_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer2_1_bn3_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer2_1_bn3_running_var',
                                               use_global_stats=False, name='x2paddle_376')
        x2paddle_377 = fluid.layers.elementwise_add(x=x2paddle_376, y=x2paddle_368, name='x2paddle_377')
        x2paddle_378 = fluid.layers.relu(x2paddle_377, name='x2paddle_378')
        x2paddle_379 = fluid.layers.conv2d(x2paddle_378, num_filters=256, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer2_2_conv1_weight', name='x2paddle_379',
                                           bias_attr=False)
        x2paddle_380 = fluid.layers.batch_norm(x2paddle_379, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer2_2_bn1_weight',
                                               bias_attr='x2paddle_fdresnext_layer2_2_bn1_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer2_2_bn1_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer2_2_bn1_running_var',
                                               use_global_stats=False, name='x2paddle_380')
        x2paddle_381 = fluid.layers.relu(x2paddle_380, name='x2paddle_381')
        x2paddle_382 = fluid.layers.conv2d(x2paddle_381, num_filters=256, filter_size=[3, 3], stride=[1, 1],
                                           padding=[1, 1], dilation=[1, 1], groups=32,
                                           param_attr='x2paddle_fdresnext_layer2_2_conv2_weight', name='x2paddle_382',
                                           bias_attr=False)
        x2paddle_383 = fluid.layers.batch_norm(x2paddle_382, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer2_2_bn2_weight',
                                               bias_attr='x2paddle_fdresnext_layer2_2_bn2_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer2_2_bn2_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer2_2_bn2_running_var',
                                               use_global_stats=False, name='x2paddle_383')
        x2paddle_384 = fluid.layers.relu(x2paddle_383, name='x2paddle_384')
        x2paddle_385 = fluid.layers.conv2d(x2paddle_384, num_filters=512, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer2_2_conv3_weight', name='x2paddle_385',
                                           bias_attr=False)
        x2paddle_386 = fluid.layers.batch_norm(x2paddle_385, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer2_2_bn3_weight',
                                               bias_attr='x2paddle_fdresnext_layer2_2_bn3_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer2_2_bn3_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer2_2_bn3_running_var',
                                               use_global_stats=False, name='x2paddle_386')
        x2paddle_387 = fluid.layers.elementwise_add(x=x2paddle_386, y=x2paddle_378, name='x2paddle_387')
        x2paddle_388 = fluid.layers.relu(x2paddle_387, name='x2paddle_388')
        x2paddle_389 = fluid.layers.conv2d(x2paddle_388, num_filters=256, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer2_3_conv1_weight', name='x2paddle_389',
                                           bias_attr=False)
        x2paddle_390 = fluid.layers.batch_norm(x2paddle_389, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer2_3_bn1_weight',
                                               bias_attr='x2paddle_fdresnext_layer2_3_bn1_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer2_3_bn1_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer2_3_bn1_running_var',
                                               use_global_stats=False, name='x2paddle_390')
        x2paddle_391 = fluid.layers.relu(x2paddle_390, name='x2paddle_391')
        x2paddle_392 = fluid.layers.conv2d(x2paddle_391, num_filters=256, filter_size=[3, 3], stride=[1, 1],
                                           padding=[1, 1], dilation=[1, 1], groups=32,
                                           param_attr='x2paddle_fdresnext_layer2_3_conv2_weight', name='x2paddle_392',
                                           bias_attr=False)
        x2paddle_393 = fluid.layers.batch_norm(x2paddle_392, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer2_3_bn2_weight',
                                               bias_attr='x2paddle_fdresnext_layer2_3_bn2_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer2_3_bn2_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer2_3_bn2_running_var',
                                               use_global_stats=False, name='x2paddle_393')
        x2paddle_394 = fluid.layers.relu(x2paddle_393, name='x2paddle_394')
        x2paddle_395 = fluid.layers.conv2d(x2paddle_394, num_filters=512, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer2_3_conv3_weight', name='x2paddle_395',
                                           bias_attr=False)
        x2paddle_396 = fluid.layers.batch_norm(x2paddle_395, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer2_3_bn3_weight',
                                               bias_attr='x2paddle_fdresnext_layer2_3_bn3_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer2_3_bn3_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer2_3_bn3_running_var',
                                               use_global_stats=False, name='x2paddle_396')
        x2paddle_397 = fluid.layers.elementwise_add(x=x2paddle_396, y=x2paddle_388, name='x2paddle_397')
        x2paddle_398 = fluid.layers.relu(x2paddle_397, name='x2paddle_398')
        x2paddle_399 = fluid.layers.conv2d(x2paddle_398, num_filters=512, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer3_0_conv1_weight', name='x2paddle_399',
                                           bias_attr=False)
        x2paddle_407 = fluid.layers.conv2d(x2paddle_398, num_filters=1024, filter_size=[1, 1], stride=[2, 2],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer3_0_downsample_0_weight',
                                           name='x2paddle_407', bias_attr=False)
        x2paddle_400 = fluid.layers.batch_norm(x2paddle_399, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer3_0_bn1_weight',
                                               bias_attr='x2paddle_fdresnext_layer3_0_bn1_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer3_0_bn1_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer3_0_bn1_running_var',
                                               use_global_stats=False, name='x2paddle_400')
        x2paddle_408 = fluid.layers.batch_norm(x2paddle_407, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer3_0_downsample_1_weight',
                                               bias_attr='x2paddle_fdresnext_layer3_0_downsample_1_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer3_0_downsample_1_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer3_0_downsample_1_running_var',
                                               use_global_stats=False, name='x2paddle_408')
        x2paddle_401 = fluid.layers.relu(x2paddle_400, name='x2paddle_401')
        x2paddle_402 = fluid.layers.conv2d(x2paddle_401, num_filters=512, filter_size=[3, 3], stride=[2, 2],
                                           padding=[1, 1], dilation=[1, 1], groups=32,
                                           param_attr='x2paddle_fdresnext_layer3_0_conv2_weight', name='x2paddle_402',
                                           bias_attr=False)
        x2paddle_403 = fluid.layers.batch_norm(x2paddle_402, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer3_0_bn2_weight',
                                               bias_attr='x2paddle_fdresnext_layer3_0_bn2_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer3_0_bn2_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer3_0_bn2_running_var',
                                               use_global_stats=False, name='x2paddle_403')
        x2paddle_404 = fluid.layers.relu(x2paddle_403, name='x2paddle_404')
        x2paddle_405 = fluid.layers.conv2d(x2paddle_404, num_filters=1024, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer3_0_conv3_weight', name='x2paddle_405',
                                           bias_attr=False)
        x2paddle_406 = fluid.layers.batch_norm(x2paddle_405, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer3_0_bn3_weight',
                                               bias_attr='x2paddle_fdresnext_layer3_0_bn3_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer3_0_bn3_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer3_0_bn3_running_var',
                                               use_global_stats=False, name='x2paddle_406')
        x2paddle_409 = fluid.layers.elementwise_add(x=x2paddle_406, y=x2paddle_408, name='x2paddle_409')
        x2paddle_410 = fluid.layers.relu(x2paddle_409, name='x2paddle_410')
        x2paddle_411 = fluid.layers.conv2d(x2paddle_410, num_filters=512, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer3_1_conv1_weight', name='x2paddle_411',
                                           bias_attr=False)
        x2paddle_412 = fluid.layers.batch_norm(x2paddle_411, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer3_1_bn1_weight',
                                               bias_attr='x2paddle_fdresnext_layer3_1_bn1_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer3_1_bn1_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer3_1_bn1_running_var',
                                               use_global_stats=False, name='x2paddle_412')
        x2paddle_413 = fluid.layers.relu(x2paddle_412, name='x2paddle_413')
        x2paddle_414 = fluid.layers.conv2d(x2paddle_413, num_filters=512, filter_size=[3, 3], stride=[1, 1],
                                           padding=[1, 1], dilation=[1, 1], groups=32,
                                           param_attr='x2paddle_fdresnext_layer3_1_conv2_weight', name='x2paddle_414',
                                           bias_attr=False)
        x2paddle_415 = fluid.layers.batch_norm(x2paddle_414, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer3_1_bn2_weight',
                                               bias_attr='x2paddle_fdresnext_layer3_1_bn2_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer3_1_bn2_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer3_1_bn2_running_var',
                                               use_global_stats=False, name='x2paddle_415')
        x2paddle_416 = fluid.layers.relu(x2paddle_415, name='x2paddle_416')
        x2paddle_417 = fluid.layers.conv2d(x2paddle_416, num_filters=1024, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer3_1_conv3_weight', name='x2paddle_417',
                                           bias_attr=False)
        x2paddle_418 = fluid.layers.batch_norm(x2paddle_417, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer3_1_bn3_weight',
                                               bias_attr='x2paddle_fdresnext_layer3_1_bn3_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer3_1_bn3_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer3_1_bn3_running_var',
                                               use_global_stats=False, name='x2paddle_418')
        x2paddle_419 = fluid.layers.elementwise_add(x=x2paddle_418, y=x2paddle_410, name='x2paddle_419')
        x2paddle_420 = fluid.layers.relu(x2paddle_419, name='x2paddle_420')
        x2paddle_421 = fluid.layers.conv2d(x2paddle_420, num_filters=512, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer3_2_conv1_weight', name='x2paddle_421',
                                           bias_attr=False)
        x2paddle_422 = fluid.layers.batch_norm(x2paddle_421, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer3_2_bn1_weight',
                                               bias_attr='x2paddle_fdresnext_layer3_2_bn1_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer3_2_bn1_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer3_2_bn1_running_var',
                                               use_global_stats=False, name='x2paddle_422')
        x2paddle_423 = fluid.layers.relu(x2paddle_422, name='x2paddle_423')
        x2paddle_424 = fluid.layers.conv2d(x2paddle_423, num_filters=512, filter_size=[3, 3], stride=[1, 1],
                                           padding=[1, 1], dilation=[1, 1], groups=32,
                                           param_attr='x2paddle_fdresnext_layer3_2_conv2_weight', name='x2paddle_424',
                                           bias_attr=False)
        x2paddle_425 = fluid.layers.batch_norm(x2paddle_424, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer3_2_bn2_weight',
                                               bias_attr='x2paddle_fdresnext_layer3_2_bn2_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer3_2_bn2_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer3_2_bn2_running_var',
                                               use_global_stats=False, name='x2paddle_425')
        x2paddle_426 = fluid.layers.relu(x2paddle_425, name='x2paddle_426')
        x2paddle_427 = fluid.layers.conv2d(x2paddle_426, num_filters=1024, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer3_2_conv3_weight', name='x2paddle_427',
                                           bias_attr=False)
        x2paddle_428 = fluid.layers.batch_norm(x2paddle_427, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer3_2_bn3_weight',
                                               bias_attr='x2paddle_fdresnext_layer3_2_bn3_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer3_2_bn3_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer3_2_bn3_running_var',
                                               use_global_stats=False, name='x2paddle_428')
        x2paddle_429 = fluid.layers.elementwise_add(x=x2paddle_428, y=x2paddle_420, name='x2paddle_429')
        x2paddle_430 = fluid.layers.relu(x2paddle_429, name='x2paddle_430')
        x2paddle_431 = fluid.layers.conv2d(x2paddle_430, num_filters=512, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer3_3_conv1_weight', name='x2paddle_431',
                                           bias_attr=False)
        x2paddle_432 = fluid.layers.batch_norm(x2paddle_431, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer3_3_bn1_weight',
                                               bias_attr='x2paddle_fdresnext_layer3_3_bn1_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer3_3_bn1_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer3_3_bn1_running_var',
                                               use_global_stats=False, name='x2paddle_432')
        x2paddle_433 = fluid.layers.relu(x2paddle_432, name='x2paddle_433')
        x2paddle_434 = fluid.layers.conv2d(x2paddle_433, num_filters=512, filter_size=[3, 3], stride=[1, 1],
                                           padding=[1, 1], dilation=[1, 1], groups=32,
                                           param_attr='x2paddle_fdresnext_layer3_3_conv2_weight', name='x2paddle_434',
                                           bias_attr=False)
        x2paddle_435 = fluid.layers.batch_norm(x2paddle_434, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer3_3_bn2_weight',
                                               bias_attr='x2paddle_fdresnext_layer3_3_bn2_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer3_3_bn2_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer3_3_bn2_running_var',
                                               use_global_stats=False, name='x2paddle_435')
        x2paddle_436 = fluid.layers.relu(x2paddle_435, name='x2paddle_436')
        x2paddle_437 = fluid.layers.conv2d(x2paddle_436, num_filters=1024, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer3_3_conv3_weight', name='x2paddle_437',
                                           bias_attr=False)
        x2paddle_438 = fluid.layers.batch_norm(x2paddle_437, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer3_3_bn3_weight',
                                               bias_attr='x2paddle_fdresnext_layer3_3_bn3_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer3_3_bn3_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer3_3_bn3_running_var',
                                               use_global_stats=False, name='x2paddle_438')
        x2paddle_439 = fluid.layers.elementwise_add(x=x2paddle_438, y=x2paddle_430, name='x2paddle_439')
        x2paddle_440 = fluid.layers.relu(x2paddle_439, name='x2paddle_440')
        x2paddle_441 = fluid.layers.conv2d(x2paddle_440, num_filters=512, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer3_4_conv1_weight', name='x2paddle_441',
                                           bias_attr=False)
        x2paddle_442 = fluid.layers.batch_norm(x2paddle_441, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer3_4_bn1_weight',
                                               bias_attr='x2paddle_fdresnext_layer3_4_bn1_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer3_4_bn1_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer3_4_bn1_running_var',
                                               use_global_stats=False, name='x2paddle_442')
        x2paddle_443 = fluid.layers.relu(x2paddle_442, name='x2paddle_443')
        x2paddle_444 = fluid.layers.conv2d(x2paddle_443, num_filters=512, filter_size=[3, 3], stride=[1, 1],
                                           padding=[1, 1], dilation=[1, 1], groups=32,
                                           param_attr='x2paddle_fdresnext_layer3_4_conv2_weight', name='x2paddle_444',
                                           bias_attr=False)
        x2paddle_445 = fluid.layers.batch_norm(x2paddle_444, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer3_4_bn2_weight',
                                               bias_attr='x2paddle_fdresnext_layer3_4_bn2_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer3_4_bn2_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer3_4_bn2_running_var',
                                               use_global_stats=False, name='x2paddle_445')
        x2paddle_446 = fluid.layers.relu(x2paddle_445, name='x2paddle_446')
        x2paddle_447 = fluid.layers.conv2d(x2paddle_446, num_filters=1024, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer3_4_conv3_weight', name='x2paddle_447',
                                           bias_attr=False)
        x2paddle_448 = fluid.layers.batch_norm(x2paddle_447, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer3_4_bn3_weight',
                                               bias_attr='x2paddle_fdresnext_layer3_4_bn3_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer3_4_bn3_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer3_4_bn3_running_var',
                                               use_global_stats=False, name='x2paddle_448')
        x2paddle_449 = fluid.layers.elementwise_add(x=x2paddle_448, y=x2paddle_440, name='x2paddle_449')
        x2paddle_450 = fluid.layers.relu(x2paddle_449, name='x2paddle_450')
        x2paddle_451 = fluid.layers.conv2d(x2paddle_450, num_filters=512, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer3_5_conv1_weight', name='x2paddle_451',
                                           bias_attr=False)
        x2paddle_452 = fluid.layers.batch_norm(x2paddle_451, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer3_5_bn1_weight',
                                               bias_attr='x2paddle_fdresnext_layer3_5_bn1_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer3_5_bn1_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer3_5_bn1_running_var',
                                               use_global_stats=False, name='x2paddle_452')
        x2paddle_453 = fluid.layers.relu(x2paddle_452, name='x2paddle_453')
        x2paddle_454 = fluid.layers.conv2d(x2paddle_453, num_filters=512, filter_size=[3, 3], stride=[1, 1],
                                           padding=[1, 1], dilation=[1, 1], groups=32,
                                           param_attr='x2paddle_fdresnext_layer3_5_conv2_weight', name='x2paddle_454',
                                           bias_attr=False)
        x2paddle_455 = fluid.layers.batch_norm(x2paddle_454, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer3_5_bn2_weight',
                                               bias_attr='x2paddle_fdresnext_layer3_5_bn2_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer3_5_bn2_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer3_5_bn2_running_var',
                                               use_global_stats=False, name='x2paddle_455')
        x2paddle_456 = fluid.layers.relu(x2paddle_455, name='x2paddle_456')
        x2paddle_457 = fluid.layers.conv2d(x2paddle_456, num_filters=1024, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer3_5_conv3_weight', name='x2paddle_457',
                                           bias_attr=False)
        x2paddle_458 = fluid.layers.batch_norm(x2paddle_457, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer3_5_bn3_weight',
                                               bias_attr='x2paddle_fdresnext_layer3_5_bn3_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer3_5_bn3_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer3_5_bn3_running_var',
                                               use_global_stats=False, name='x2paddle_458')
        x2paddle_459 = fluid.layers.elementwise_add(x=x2paddle_458, y=x2paddle_450, name='x2paddle_459')
        x2paddle_460 = fluid.layers.relu(x2paddle_459, name='x2paddle_460')
        x2paddle_461 = fluid.layers.conv2d(x2paddle_460, num_filters=1024, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer4_0_conv1_weight', name='x2paddle_461',
                                           bias_attr=False)
        x2paddle_469 = fluid.layers.conv2d(x2paddle_460, num_filters=2048, filter_size=[1, 1], stride=[2, 2],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer4_0_downsample_0_weight',
                                           name='x2paddle_469', bias_attr=False)
        x2paddle_462 = fluid.layers.batch_norm(x2paddle_461, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer4_0_bn1_weight',
                                               bias_attr='x2paddle_fdresnext_layer4_0_bn1_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer4_0_bn1_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer4_0_bn1_running_var',
                                               use_global_stats=False, name='x2paddle_462')
        x2paddle_470 = fluid.layers.batch_norm(x2paddle_469, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer4_0_downsample_1_weight',
                                               bias_attr='x2paddle_fdresnext_layer4_0_downsample_1_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer4_0_downsample_1_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer4_0_downsample_1_running_var',
                                               use_global_stats=False, name='x2paddle_470')
        x2paddle_463 = fluid.layers.relu(x2paddle_462, name='x2paddle_463')
        x2paddle_464 = fluid.layers.conv2d(x2paddle_463, num_filters=1024, filter_size=[3, 3], stride=[2, 2],
                                           padding=[1, 1], dilation=[1, 1], groups=32,
                                           param_attr='x2paddle_fdresnext_layer4_0_conv2_weight', name='x2paddle_464',
                                           bias_attr=False)
        x2paddle_465 = fluid.layers.batch_norm(x2paddle_464, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer4_0_bn2_weight',
                                               bias_attr='x2paddle_fdresnext_layer4_0_bn2_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer4_0_bn2_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer4_0_bn2_running_var',
                                               use_global_stats=False, name='x2paddle_465')
        x2paddle_466 = fluid.layers.relu(x2paddle_465, name='x2paddle_466')
        x2paddle_467 = fluid.layers.conv2d(x2paddle_466, num_filters=2048, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer4_0_conv3_weight', name='x2paddle_467',
                                           bias_attr=False)
        x2paddle_468 = fluid.layers.batch_norm(x2paddle_467, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer4_0_bn3_weight',
                                               bias_attr='x2paddle_fdresnext_layer4_0_bn3_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer4_0_bn3_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer4_0_bn3_running_var',
                                               use_global_stats=False, name='x2paddle_468')
        x2paddle_471 = fluid.layers.elementwise_add(x=x2paddle_468, y=x2paddle_470, name='x2paddle_471')
        x2paddle_472 = fluid.layers.relu(x2paddle_471, name='x2paddle_472')
        x2paddle_473 = fluid.layers.conv2d(x2paddle_472, num_filters=1024, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer4_1_conv1_weight', name='x2paddle_473',
                                           bias_attr=False)
        x2paddle_474 = fluid.layers.batch_norm(x2paddle_473, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer4_1_bn1_weight',
                                               bias_attr='x2paddle_fdresnext_layer4_1_bn1_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer4_1_bn1_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer4_1_bn1_running_var',
                                               use_global_stats=False, name='x2paddle_474')
        x2paddle_475 = fluid.layers.relu(x2paddle_474, name='x2paddle_475')
        x2paddle_476 = fluid.layers.conv2d(x2paddle_475, num_filters=1024, filter_size=[3, 3], stride=[1, 1],
                                           padding=[1, 1], dilation=[1, 1], groups=32,
                                           param_attr='x2paddle_fdresnext_layer4_1_conv2_weight', name='x2paddle_476',
                                           bias_attr=False)
        x2paddle_477 = fluid.layers.batch_norm(x2paddle_476, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer4_1_bn2_weight',
                                               bias_attr='x2paddle_fdresnext_layer4_1_bn2_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer4_1_bn2_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer4_1_bn2_running_var',
                                               use_global_stats=False, name='x2paddle_477')
        x2paddle_478 = fluid.layers.relu(x2paddle_477, name='x2paddle_478')
        x2paddle_479 = fluid.layers.conv2d(x2paddle_478, num_filters=2048, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer4_1_conv3_weight', name='x2paddle_479',
                                           bias_attr=False)
        x2paddle_480 = fluid.layers.batch_norm(x2paddle_479, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer4_1_bn3_weight',
                                               bias_attr='x2paddle_fdresnext_layer4_1_bn3_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer4_1_bn3_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer4_1_bn3_running_var',
                                               use_global_stats=False, name='x2paddle_480')
        x2paddle_481 = fluid.layers.elementwise_add(x=x2paddle_480, y=x2paddle_472, name='x2paddle_481')
        x2paddle_482 = fluid.layers.relu(x2paddle_481, name='x2paddle_482')
        x2paddle_483 = fluid.layers.conv2d(x2paddle_482, num_filters=1024, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer4_2_conv1_weight', name='x2paddle_483',
                                           bias_attr=False)
        x2paddle_484 = fluid.layers.batch_norm(x2paddle_483, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer4_2_bn1_weight',
                                               bias_attr='x2paddle_fdresnext_layer4_2_bn1_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer4_2_bn1_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer4_2_bn1_running_var',
                                               use_global_stats=False, name='x2paddle_484')
        x2paddle_485 = fluid.layers.relu(x2paddle_484, name='x2paddle_485')
        x2paddle_486 = fluid.layers.conv2d(x2paddle_485, num_filters=1024, filter_size=[3, 3], stride=[1, 1],
                                           padding=[1, 1], dilation=[1, 1], groups=32,
                                           param_attr='x2paddle_fdresnext_layer4_2_conv2_weight', name='x2paddle_486',
                                           bias_attr=False)
        x2paddle_487 = fluid.layers.batch_norm(x2paddle_486, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer4_2_bn2_weight',
                                               bias_attr='x2paddle_fdresnext_layer4_2_bn2_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer4_2_bn2_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer4_2_bn2_running_var',
                                               use_global_stats=False, name='x2paddle_487')
        x2paddle_488 = fluid.layers.relu(x2paddle_487, name='x2paddle_488')
        x2paddle_489 = fluid.layers.conv2d(x2paddle_488, num_filters=2048, filter_size=[1, 1], stride=[1, 1],
                                           padding=[0, 0], dilation=[1, 1], groups=1,
                                           param_attr='x2paddle_fdresnext_layer4_2_conv3_weight', name='x2paddle_489',
                                           bias_attr=False)
        x2paddle_490 = fluid.layers.batch_norm(x2paddle_489, momentum=0.8999999761581421, epsilon=9.999999747378752e-06,
                                               data_layout='NCHW', is_test=True,
                                               param_attr='x2paddle_fdresnext_layer4_2_bn3_weight',
                                               bias_attr='x2paddle_fdresnext_layer4_2_bn3_bias',
                                               moving_mean_name='x2paddle_fdresnext_layer4_2_bn3_running_mean',
                                               moving_variance_name='x2paddle_fdresnext_layer4_2_bn3_running_var',
                                               use_global_stats=False, name='x2paddle_490')
        x2paddle_491 = fluid.layers.elementwise_add(x=x2paddle_490, y=x2paddle_482, name='x2paddle_491')
        x2paddle_492 = fluid.layers.relu(x2paddle_491, name='x2paddle_492')
        x2paddle_493 = fluid.layers.pool2d(x2paddle_492, pool_type='avg', global_pooling=True, name='x2paddle_493')
        x2paddle_494 = fluid.layers.flatten(x2paddle_493, axis=1, name='x2paddle_494')
        x2paddle_output_mm = fluid.layers.matmul(x=x2paddle_494, y=x2paddle_fdresnext_fc_weight, transpose_x=False,
                                                 transpose_y=True, alpha=1.0, name='x2paddle_output_mm')
        x2paddle_output = fluid.layers.elementwise_add(x=x2paddle_output_mm, y=x2paddle_fdresnext_fc_bias,
                                                       name='x2paddle_output')

        return x2paddle_output


def fdresnext():
    return Fdresnext()