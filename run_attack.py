# coding = utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import functools
import paddle.fluid as fluid

import models
from attack.attacks import MIFGSM
from utils import init_prog, save_adv_image, process_img, calc_mse, add_arguments, print_arguments, tensor2img_floor, tensor2img

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

add_arg('class_dim', int, 121, "Class number.")
add_arg('shape', str, "3,224,224", "output image shape")
add_arg('input', str, "./input_images/", "Input directory with images")
add_arg('output', str, "./result/", "Output directory with images")

args = parser.parse_args()
print_arguments(args)

image_shape = [int(m) for m in args.shape.split(",")]
class_dim = args.class_dim
input_dir = args.input
output_dir = args.output
model_name_resnet = 'ResNeXt50_32x4d'
model_name_mobilenet = 'MobileNetV2_x2_0'
pretrained_model = './models_parameters/'

val_list = 'val_list.txt'
use_gpu = True

adv_program = fluid.Program()


def input_diversity(input_layer, aug, pad, rotate, random_size, flip, noise):
    input_layer = input_layer * noise
    aug_grid = fluid.layers.affine_grid(aug, out_shape=input_layer.shape)
    aug_input = fluid.layers.grid_sampler(input_layer, grid=aug_grid)

    rotate_grid = fluid.layers.affine_grid(rotate, out_shape=aug_input.shape)
    rotate_input = fluid.layers.grid_sampler(aug_input, grid=rotate_grid)

    flip_grid = fluid.layers.affine_grid(flip, out_shape=rotate_input.shape)
    flip_input = fluid.layers.grid_sampler(rotate_input, grid=flip_grid)

    resized_input = fluid.layers.resize_nearest(flip_input, out_shape=[random_size, random_size])
    padded_input = fluid.layers.pad2d(resized_input, pad, pad_value=0.0)
    return padded_input


with fluid.program_guard(adv_program):
    input_layer = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    input_layer.stop_gradient = False
    random_size = fluid.layers.data(name='size', shape=[1 * 6], dtype='int32', append_batch_size=False)
    pad = fluid.layers.data(name='pad', shape=[4 * 6], dtype='int32', append_batch_size=False)
    aug = fluid.layers.data('aug', shape=[2 * 6, 3], dtype='float32')
    rotate = fluid.layers.data('rotate', shape=[2 * 6, 3], dtype='float32')
    flip = fluid.layers.data('flip', shape=[2 * 6, 3], dtype='float32')
    weight = fluid.layers.data('weight', shape=[6, 121], dtype='float32')
    noise = fluid.layers.data('noise', shape=[6 * 3], dtype='float32')

    size_0, size_1, size_2, size_3, size_4, size_5 = fluid.layers.split(random_size, num_or_sections=6, dim=0)
    pad_0, pad_1, pad_2, pad_3, pad_4, pad_5 = fluid.layers.split(pad, num_or_sections=6, dim=0)
    aug_0, aug_1, aug_2, aug_3, aug_4, aug_5 = fluid.layers.split(aug, num_or_sections=6, dim=1)
    rotate_0, rotate_1, rotate_2, rotate_3, rotate_4, rotate_5 = fluid.layers.split(rotate, num_or_sections=6, dim=1)
    flip_0, flip_1, flip_2, flip_3, flip_4, flip_5 = fluid.layers.split(flip, num_or_sections=6, dim=1)
    weight_0, weight_1, weight_2, weight_3, weight_4, weight_5 = fluid.layers.split(weight, num_or_sections=6, dim=1)
    noise_0, noise_1, noise_2, noise_3, noise_4, noise_5 = fluid.layers.split(noise, num_or_sections=6, dim=1)

    weight_0 = fluid.layers.squeeze(weight_0, axes=[1])
    weight_1 = fluid.layers.squeeze(weight_1, axes=[1])
    weight_2 = fluid.layers.squeeze(weight_2, axes=[1])
    weight_3 = fluid.layers.squeeze(weight_3, axes=[1])
    weight_4 = fluid.layers.squeeze(weight_4, axes=[1])
    weight_5 = fluid.layers.squeeze(weight_5, axes=[1])

    model_resnet = models.__dict__[model_name_resnet]()
    out_logits_resnet = model_resnet.net(input=input_diversity(input_layer, aug_0, pad_0, rotate_0, size_0, flip_0, noise_0),
                                         class_dim=class_dim)

    model_mobilenet = models.__dict__[model_name_mobilenet]()
    out_logits_mobilenet = model_mobilenet.net(
        input=input_diversity(input_layer, aug_1, pad_1, rotate_1, size_1, flip_1, noise_1), class_dim=class_dim)

    model_densenet = models.__dict__['densenet']()
    out_logits_densenet = model_densenet.net(input_diversity(input_layer, aug_2, pad_2, rotate_2, size_2, flip_2, noise_2))

    model_wideresnet = models.__dict__['wideresnet']()
    out_logits_wideresnet = model_wideresnet.net(input_diversity(input_layer, aug_3, pad_3, rotate_3, size_3, flip_3, noise_3))

    model_fddensenet = models.__dict__['fddensenet']()
    out_logits_fddensenet = model_fddensenet.net(input_diversity(input_layer, aug_4, pad_4, rotate_4, size_4, flip_4, noise_4))

    model_fdresnext = models.__dict__['fdresnext']()
    out_logits_fdresnext = model_fdresnext.net(input_diversity(input_layer, aug_5, pad_5, rotate_5, size_5, flip_5, noise_5))

    out_logits = (out_logits_resnet * weight_0 + out_logits_mobilenet * weight_1 + out_logits_densenet * weight_2
                  + weight_3 * out_logits_wideresnet + 2 * weight_4 * out_logits_fddensenet + 2 * weight_5 * out_logits_fdresnext) \
                 / (weight_0 + weight_1 + weight_2 + weight_3 + 2 * weight_4 + 2 * weight_5)
    out = fluid.layers.softmax(out_logits)

    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    fluid.io.load_persistables(exe, pretrained_model)

init_prog(adv_program)
eval_program = adv_program.clone(for_test=True)

with fluid.program_guard(adv_program):
    label = fluid.layers.data(name="label", shape=[1], dtype='int64')  # gt label
    loss = fluid.layers.cross_entropy(input=out, label=label)
    gradients = fluid.backward.gradients(targets=loss, inputs=[input_layer])[0]


def attack_driver(img, src_label, filename):
    eps = 16. / 255.
    step = eps / 50
    iteration = 300
    imgname, ext = os.path.splitext(filename)

    adv = MIFGSM(adv_program=adv_program, o=img,
                 input_layer=input_layer, step_size=step, epsilon=eps,
                 gt_label=src_label, use_gpu=use_gpu,
                 iteration=iteration, gradients=gradients, imgname=imgname)

    print("{1}\ttarget class {0}".format(src_label, filename))
    adv_img = tensor2img(adv)
    return adv_img


def get_original_file(filepath):
    with open(filepath, 'r') as cfile:
        full_lines = [line.strip() for line in cfile]
    cfile.close()
    original_files = []
    for line in full_lines:
        label, file_name = line.split()
        original_files.append([file_name, int(label)])
    return original_files


def gen_adv():
    mse = 0
    original_files = get_original_file(input_dir + val_list)

    for filename, gt_label in original_files:
        img_path = input_dir + filename
        img = process_img(img_path)

        image_name, image_ext = filename.split('.')
        adv_img = attack_driver(img, gt_label, filename)
        save_adv_image(adv_img, output_dir + image_name + '.png')
        org_img = tensor2img(img)

        score = calc_mse(org_img, adv_img)
        print(score)
        mse += score
    print("ADV {} files, AVG MSE: {} ".format(len(original_files), mse / len(original_files)))


if __name__ == '__main__':
    gen_adv()
