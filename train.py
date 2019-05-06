from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import time
import sys
import functools
import math
import paddle
import paddle.fluid as fluid
import paddle.dataset.flowers as flowers
import models
import reader
import argparse
import functools
import subprocess
import utils
from utils.learning_rate import cosine_decay
from utility import add_arguments, print_arguments
import models
import models_name
import paddle.fluid.profiler as profiler

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,   1,                  "Minibatch size.")
add_arg('use_gpu',          bool,  True,                 "Whether to use GPU or not.")
add_arg('total_images',     int,   1281167,              "Training image number.")
add_arg('num_epochs',       int,   120,                  "number of epochs.")
add_arg('class_dim',        int,   1000,                 "Class number.")
add_arg('image_shape',      str,   "3,224,224",          "input image size")
add_arg('model_save_dir',   str,   "output",             "model save directory")
add_arg('with_mem_opt',     bool,  False,                 "Whether to use memory optimization or not.")
add_arg('pretrained_model', str,   None,                 "Whether to use pretrained model.")
add_arg('checkpoint',       str,   None,                 "Whether to resume checkpoint.")
add_arg('lr',               float, 0.1,                  "set learning rate.")
#add_arg('lr_strategy',      str,   "piecewise_decay",    "Set the learning rate decay strategy.")
add_arg('lr_strategy',      str,   "",    "Set the learning rate decay strategy.")
add_arg('model',            str,   "SE_ResNeXt50_32x4d", "Set the network to use.")
add_arg('enable_ce',        bool,  False,                "If set True, enable continuous evaluation job.")
add_arg('data_dir',         str,   "./data/ILSVRC2012",  "The ImageNet dataset root dir.")
add_arg('model_category',   str,   "models",             "Whether to use models_name or not, valid value:'models','models_name'" )
add_arg('run_mode',         str,   "train",              "train, infer, fused_infer.")
add_arg('precision',        str,   "int16",              "int16, int8")
add_arg('place',            str,   "cuda",               "cuda, xsim.")
# yapf: enabl


def set_models(model):
    global models
    if model == "models":
        models = models
    else:
        models = models_name


def optimizer_setting(params):
    ls = params["learning_strategy"]
    if ls["name"] == "piecewise_decay":
        if "total_images" not in params:
            total_images = 1281167
        else:
            total_images = params["total_images"]
        batch_size = ls["batch_size"]
        step = int(total_images / batch_size + 1)

        bd = [step * e for e in ls["epochs"]]
        base_lr = params["lr"]
        lr = []
        lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.piecewise_decay(
                boundaries=bd, values=lr),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))

    elif ls["name"] == "cosine_decay":
        if "total_images" not in params:
            total_images = 1281167
        else:
            total_images = params["total_images"]

        batch_size = ls["batch_size"]
        step = int(total_images / batch_size + 1)

        lr = params["lr"]
        num_epochs = params["num_epochs"]

        optimizer = fluid.optimizer.Momentum(
            learning_rate=cosine_decay(
                learning_rate=lr, step_each_epoch=step, epochs=num_epochs),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(4e-5))
    elif ls["name"] == "exponential_decay":
        if "total_images" not in params:
            total_images = 1281167
        else:
            total_images = params["total_images"]
        batch_size = ls["batch_size"]
        step = int(total_images / batch_size +1)
        lr = params["lr"]
        num_epochs = params["num_epochs"]
        learning_decay_rate_factor=ls["learning_decay_rate_factor"]
        num_epochs_per_decay = ls["num_epochs_per_decay"]
        NUM_GPUS = 1

        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.exponential_decay(
                learning_rate = lr * NUM_GPUS,
                decay_steps = step * num_epochs_per_decay / NUM_GPUS,
                decay_rate = learning_decay_rate_factor),
            momentum=0.9,

            regularization = fluid.regularizer.L2Decay(4e-5))

    else:
        lr = params["lr"]
        """
        optimizer = fluid.optimizer.Momentum(
            learning_rate=lr,
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))
        """
        optimizer = fluid.optimizer.Momentum(
            learning_rate=lr,
            momentum=0.9)

    return optimizer

def net_config(main_prog, image, label, model, args):
    model_list = [m for m in dir(models) if "__" not in m]
    assert args.model in model_list,"{} is not lists: {}".format(
        args.model, model_list)

    class_dim = args.class_dim
    model_name = args.model

    if args.enable_ce:
        assert model_name == "SE_ResNeXt50_32x4d"
        model.params["dropout_seed"] = 100
        class_dim = 102

    if model_name == "GoogleNet":
        out0, out1, out2 = model.net(input=image, class_dim=class_dim)

        infer_prog = main_prog.clone(for_test=True)

        cost0 = fluid.layers.cross_entropy(input=out0, label=label)
        cost1 = fluid.layers.cross_entropy(input=out1, label=label)
        cost2 = fluid.layers.cross_entropy(input=out2, label=label)
        avg_cost0 = fluid.layers.mean(x=cost0)
        avg_cost1 = fluid.layers.mean(x=cost1)
        avg_cost2 = fluid.layers.mean(x=cost2)

        avg_cost = avg_cost0 + 0.3 * avg_cost1 + 0.3 * avg_cost2
        acc_top1 = fluid.layers.accuracy(input=out0, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out0, label=label, k=5)

        return infer_prog, out0, avg_cost, acc_top1, acc_top5
    else:
        out = model.net(input=image, class_dim=class_dim)

        infer_prog = main_prog.clone(for_test=True)

        cost = fluid.layers.cross_entropy(input=out, label=label)

        avg_cost = fluid.layers.mean(x=cost)
        acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5) 

        return infer_prog, out, avg_cost, acc_top1, acc_top5


def build_program(is_train, main_prog, startup_prog, args):
    image_shape = [int(m) for m in args.image_shape.split(",")]
    model_name = args.model
    model_list = [m for m in dir(models) if "__" not in m]
    assert model_name in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)
    model = models.__dict__[model_name]()
    with fluid.program_guard(main_prog, startup_prog):
        """
        py_reader = fluid.layers.py_reader(
            capacity=16,
            shapes=[[-1] + image_shape, [-1, 1]],
            lod_levels=[0, 0],
            dtypes=["float32", "int64"],
            use_double_buffer=True)
        """
        with fluid.unique_name.guard():
            #image, label = fluid.layers.read_file(py_reader)
            #avg_cost, acc_top1, acc_top5 = net_config(image, label, model, args)

            data_layer = fluid.layers.data(name='data', shape=[3, 224, 224], dtype='float32')
            label_layer = fluid.layers.data(name='label', shape=[1], dtype='int64')
            infer_prog, out, avg_cost, acc_top1, acc_top5 = net_config(main_prog, data_layer, label_layer, model, args)

            avg_cost.persistable = True
            acc_top1.persistable = True
            acc_top5.persistable = True
            if is_train:
                params = model.params
                params["total_images"] = args.total_images
                params["lr"] = args.lr
                params["num_epochs"] = args.num_epochs
                params["learning_strategy"]["batch_size"] = args.batch_size
                params["learning_strategy"]["name"] = args.lr_strategy

                optimizer = optimizer_setting(params)
                optimizer.minimize(avg_cost)
                return infer_prog, out, avg_cost, acc_top1, acc_top5

    #return py_reader, avg_cost, acc_top1, acc_top5
    return avg_cost, acc_top1, acc_top5


def train(args):
    # parameters from arguments
    model_name = args.model
    checkpoint = args.checkpoint
    pretrained_model = args.pretrained_model
    with_memory_optimization = args.with_mem_opt
    model_save_dir = args.model_save_dir

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    test_prog = fluid.Program()

    if args.enable_ce:
        startup_prog.random_seed = 1000
        train_prog.random_seed = 1000

    #train_py_reader, train_cost, train_acc1, train_acc5 = build_program(
    infer_prog, train_out, train_cost, train_acc1, train_acc5 = build_program(
        is_train=True,
        main_prog=train_prog,
        startup_prog=startup_prog,
        args=args)
    #test_py_reader, test_cost, test_acc1, test_acc5 = build_program(
    test_cost, test_acc1, test_acc5 = build_program(
        is_train=False,
        main_prog=test_prog,
        startup_prog=startup_prog,
        args=args)
    test_prog = test_prog.clone(for_test=True)

    if with_memory_optimization:
        fluid.memory_optimize(train_prog)
        fluid.memory_optimize(test_prog)

    """
    print("-------------------------------------")
    for block in train_prog.blocks:
        for op in block.ops:
            print("op_train: ", op.type)
    print("-------------------------------------")
    for block in test_prog.blocks:
        for op in block.ops:
            print("op_infer: ", op.type)
    exit()
    """

    #place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    #place = fluid.XSIMPlace()
    #place = fluid.XCPUPlace()
    if args.place == "cuda":
        place = fluid.CUDAPlace(0)
    elif args.place == "xsim":
        place = fluid.XSIMPlace()
    else:
        print("Unsurpported place!")
        exit()

    exe = fluid.Executor(place)

    print("Run startup...")
    exe.run(startup_prog)

    train_fetch_list = [train_cost.name, train_acc1.name, train_acc5.name]

    if (args.run_mode == "train"):
        prog = train_prog
    elif (args.run_mode == "infer"):
        prog = test_prog
    elif (args.run_mode == "fused_infer"):
        print("Transpiling...")
        inference_transpiler_program = test_prog.clone()
        t = fluid.transpiler.InferenceTranspiler()
        config = {
                "use_fake_max": True,
                "conv_weight_type": args.precision,
                "fc_weight_type": args.precision,
                "fc_pretrans_a": False,
                "fc_pretrans_b": True
                }
        t.transpile_xpu(inference_transpiler_program, place, config)
        prog = inference_transpiler_program
    else:
        print("bad run_mode: ", args.run_mode)
        exit()


    print("Running...")
    img_data = np.random.random([args.batch_size, 3, 224, 224]).astype('float32')
    y_data = np.random.random([args.batch_size, 1]).astype('int64')

    if args.place == "cuda":
        # warm up
        loss, acc1, acc5 = exe.run(prog,
                feed={"data": img_data, "label": y_data},
                fetch_list=train_fetch_list)

        profiler.start_profiler("All")

    loss, acc1, acc5 = exe.run(prog,
            feed={"data": img_data, "label": y_data},
            fetch_list=train_fetch_list)

    if args.place == "cuda":
        profiler.stop_profiler("total", "/tmp/profile")


def main():
    args = parser.parse_args()
    models_now = args.model_category
    assert models_now in ["models", "models_name"], "{} is not in lists: {}".format(
            models_now, ["models", "models_name"])
    set_models(models_now)
    print_arguments(args)
    train(args)


if __name__ == '__main__':
    main()
