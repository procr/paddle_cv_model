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
import reader
import argparse
import functools
import subprocess
import utils
from utils.learning_rate import cosine_decay
#from utils.fp16_utils import create_master_params_grads, master_param_to_train_param
from utility import add_arguments, print_arguments

IMAGENET1000 = 1281167

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,   256,                  "Minibatch size.")
add_arg('use_gpu',          bool,  True,                 "Whether to use GPU or not.")
add_arg('total_images',     int,   1281167,              "Training image number.")
add_arg('num_epochs',       int,   1,                    "number of epochs.")
add_arg('class_dim',        int,   1000,                  "Class number.")
add_arg('image_shape',      str,   "3,224,224",          "input image size")
add_arg('model_save_dir',   str,   "output",             "model save directory")
add_arg('with_mem_opt',     bool,  True,                 "Whether to use memory optimization or not.")
add_arg('pretrained_model', str,   None,                 "Whether to use pretrained model.")
add_arg('checkpoint',       str,   None,                 "Whether to resume checkpoint.")
add_arg('lr',               float, 0.1,                  "set learning rate.")
add_arg('lr_strategy',      str,   "",    "Set the learning rate decay strategy.")
add_arg('model',            str,   "SE_ResNeXt50_32x4d", "Set the network to use.")
add_arg('enable_ce',        bool,  False,                "If set True, enable continuous evaluation job.")
add_arg('data_dir',         str,   "./data/ILSVRC2012",  "The ImageNet dataset root dir.")
add_arg('model_category',   str,   "models",             "Whether to use models_name or not, valid value:'models','models_name'." )
add_arg('fp16',             bool,  False,                "Enable half precision training with fp16." )
add_arg('scale_loss',       float, 1.0,                  "Scale loss for fp16." )
add_arg('l2_decay',         float, 1e-4,                 "L2_decay parameter.")
add_arg('momentum_rate',    float, 0.9,                  "momentum_rate.")
add_arg('run_mode',         str,   "train",              "train, infer, fused_infer.")
add_arg('place',            str,   "cuda",               "cuda, xsim.")
# yapf: enable


def set_models(model_category):
    global models
    assert model_category in ["models", "models_name"
                              ], "{} is not in lists: {}".format(
                                  model_category, ["models", "models_name"])
    if model_category == "models_name":
        import models_name as models
    else:
        import models as models


def optimizer_setting(params):
    ls = params["learning_strategy"]
    l2_decay = params["l2_decay"]
    momentum_rate = params["momentum_rate"]
    if ls["name"] == "piecewise_decay":
        if "total_images" not in params:
            total_images = IMAGENET1000
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
            momentum=momentum_rate,
            regularization=fluid.regularizer.L2Decay(l2_decay))

    elif ls["name"] == "cosine_decay":
        if "total_images" not in params:
            total_images = IMAGENET1000
        else:
            total_images = params["total_images"]
        batch_size = ls["batch_size"]
        l2_decay = params["l2_decay"]
        momentum_rate = params["momentum_rate"]
        step = int(total_images / batch_size + 1)

        lr = params["lr"]
        num_epochs = params["num_epochs"]

        optimizer = fluid.optimizer.Momentum(
            learning_rate=cosine_decay(
                learning_rate=lr, step_each_epoch=step, epochs=num_epochs),
            momentum=momentum_rate,
            regularization=fluid.regularizer.L2Decay(l2_decay))
    elif ls["name"] == "linear_decay":
        if "total_images" not in params:
            total_images = IMAGENET1000
        else:
            total_images = params["total_images"]
        batch_size = ls["batch_size"]
        num_epochs = params["num_epochs"]
        start_lr = params["lr"]
        l2_decay = params["l2_decay"]
        momentum_rate = params["momentum_rate"]
        end_lr = 0
        total_step = int((total_images / batch_size) * num_epochs)
        lr = fluid.layers.polynomial_decay(
            start_lr, total_step, end_lr, power=1)
        optimizer = fluid.optimizer.Momentum(
            learning_rate=lr,
            momentum=momentum_rate,
            regularization=fluid.regularizer.L2Decay(l2_decay))
    else:
        lr = params["lr"]
        l2_decay = params["l2_decay"]
        momentum_rate = params["momentum_rate"]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=lr,
            momentum=momentum_rate,
            regularization=fluid.regularizer.L2Decay(l2_decay))

    return optimizer


def net_config(image, label, model, args):
    model_list = [m for m in dir(models) if "__" not in m]
    assert args.model in model_list, "{} is not lists: {}".format(args.model,
                                                                  model_list)

    class_dim = args.class_dim
    model_name = args.model

    if args.enable_ce:
        assert model_name == "SE_ResNeXt50_32x4d"
        model.params["dropout_seed"] = 100
        class_dim = 102

    if model_name == "GoogleNet":
        out0, out1, out2 = model.net(input=image, class_dim=class_dim)
        cost0 = fluid.layers.cross_entropy(input=out0, label=label)
        cost1 = fluid.layers.cross_entropy(input=out1, label=label)
        cost2 = fluid.layers.cross_entropy(input=out2, label=label)
        avg_cost0 = fluid.layers.mean(x=cost0)
        avg_cost1 = fluid.layers.mean(x=cost1)
        avg_cost2 = fluid.layers.mean(x=cost2)

        avg_cost = avg_cost0 + 0.3 * avg_cost1 + 0.3 * avg_cost2
        acc_top1 = fluid.layers.accuracy(input=out0, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out0, label=label, k=5)
    else:
        out = model.net(input=image, class_dim=class_dim)
        cost, pred = fluid.layers.softmax_with_cross_entropy(
            out, label, return_softmax=True)
        if args.scale_loss > 1:
            avg_cost = fluid.layers.mean(x=cost) * float(args.scale_loss)
        else:
            avg_cost = fluid.layers.mean(x=cost)

        acc_top1 = fluid.layers.accuracy(input=pred, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=pred, label=label, k=5)

    return avg_cost, acc_top1, acc_top5


def build_program(is_train, main_prog, startup_prog, args):
    image_shape = [int(m) for m in args.image_shape.split(",")]
    model_name = args.model
    model_list = [m for m in dir(models) if "__" not in m]
    assert model_name in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)
    model = models.__dict__[model_name]()
    with fluid.program_guard(main_prog, startup_prog):
        py_reader = fluid.layers.py_reader(
            capacity=16,
            shapes=[[-1] + image_shape, [-1, 1]],
            lod_levels=[0, 0],
            dtypes=["float32", "int64"],
            use_double_buffer=True)
        with fluid.unique_name.guard():
            image, label = fluid.layers.read_file(py_reader)
            if args.fp16:
                image = fluid.layers.cast(image, "float16")
            avg_cost, acc_top1, acc_top5 = net_config(image, label, model, args)
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
                params["l2_decay"] = args.l2_decay
                params["momentum_rate"] = args.momentum_rate

                optimizer = optimizer_setting(params)
                if args.fp16:
                    params_grads = optimizer.backward(avg_cost)
                    master_params_grads = create_master_params_grads(
                        params_grads, main_prog, startup_prog, args.scale_loss)
                    optimizer.apply_gradients(master_params_grads)
                    master_param_to_train_param(master_params_grads,
                                                params_grads, main_prog)
                else:
                    optimizer.minimize(avg_cost)
                global_lr = optimizer._global_learning_rate()

    if is_train:
        return py_reader, image.name, label.name, avg_cost, acc_top1, acc_top5, global_lr
    else:
        return py_reader, avg_cost, acc_top1, acc_top5

def train(args, model_path, place):
    # parameters from arguments
    with_memory_optimization = args.with_mem_opt

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    if args.enable_ce:
        startup_prog.random_seed = 1000
        train_prog.random_seed = 1000

    train_py_reader, data_name, label_name, train_cost, train_acc1, train_acc5, global_lr = build_program(
        is_train=True,
        main_prog=train_prog,
        startup_prog=startup_prog,
        args=args)

    if with_memory_optimization:
        fluid.memory_optimize(train_prog)

    exe = fluid.Executor(place)
    exe.run(startup_prog)

    """
    visible_device = os.getenv('CUDA_VISIBLE_DEVICES')
    if visible_device:
        device_num = len(visible_device.split(','))
    else:
        device_num = subprocess.check_output(
            ['nvidia-smi', '-L']).decode().count('\n')
    """
    device_num = 1

    train_batch_size = args.batch_size / device_num
    if not args.enable_ce:
        train_reader = paddle.batch(
            reader.train(), batch_size=train_batch_size, drop_last=True)
    else:
        # use flowers dataset for CE and set use_xmap False to avoid disorder data
        # but it is time consuming. For faster speed, need another dataset.
        import random
        random.seed(0)
        np.random.seed(0)
        train_reader = paddle.batch(
            flowers.train(use_xmap=False),
            batch_size=train_batch_size,
            drop_last=True)

    train_py_reader.decorate_paddle_reader(train_reader)
    train_exe = fluid.ParallelExecutor(
        main_program=train_prog,
        use_cuda=bool(args.use_gpu),
        loss_name=train_cost.name)

    save_target_vars = [train_cost, train_acc1, train_acc5, global_lr]
    train_fetch_list = [
        train_cost.name, train_acc1.name, train_acc5.name, global_lr.name
    ]

    params = models.__dict__[args.model]().params
    for pass_id in range(params["num_epochs"]):
        train_py_reader.start()
        train_info = [[], [], []]
        train_time = []
        batch_id = 0
        try:
            while True:
                t1 = time.time()
                loss, acc1, acc5, lr = train_exe.run(
                    fetch_list=train_fetch_list)

                t2 = time.time()
                period = t2 - t1
                loss = np.mean(np.array(loss))
                acc1 = np.mean(np.array(acc1))
                acc5 = np.mean(np.array(acc5))
                train_info[0].append(loss)
                train_info[1].append(acc1)
                train_info[2].append(acc5)
                lr = np.mean(np.array(lr))
                train_time.append(period)

                if batch_id % 10 == 0:
                    print("Pass {0}, trainbatch {1}, loss {2}, \
                        acc1 {3}, acc5 {4}, lr{5}, time {6}"
                          .format(pass_id, batch_id, loss, acc1, acc5, "%.5f" %
                                  lr, "%2.2f sec" % period))
                    sys.stdout.flush()
                batch_id += 1
        except fluid.core.EOFException:
            train_py_reader.reset()

        train_loss = np.array(train_info[0]).mean()
        train_acc1 = np.array(train_info[1]).mean()
        train_acc5 = np.array(train_info[2]).mean()
        train_speed = np.array(train_time).mean() / (train_batch_size *
                                                     device_num)

        print("End pass {0}, train_loss {1}, train_acc1 {2}, train_acc5 {3}, ".format(
                  pass_id, train_loss, train_acc1, train_acc5))
        sys.stdout.flush()

        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        fluid.io.save_inference_model(model_path, 
                [data_name, label_name],
                save_target_vars,
                exe,
                main_program=train_prog,
                model_filename="__model__",
                params_filename="__params__")
        print("save to", model_path)

def infer(args, model_path, place):
    # parameters from arguments
    with_memory_optimization = args.with_mem_opt

    startup_prog = fluid.Program()
    test_prog = fluid.Program()
    if args.enable_ce:
        startup_prog.random_seed = 1000

    test_py_reader, test_cost, test_acc1, test_acc5 = build_program(
        is_train=False,
        main_prog=test_prog,
        startup_prog=startup_prog,
        args=args)
    test_prog = test_prog.clone(for_test=True)

    if (args.run_mode == "fused_infer"):
        inference_transpiler_program = test_prog.clone()
        t = fluid.transpiler.InferenceTranspiler()
        t.transpile_xpu(inference_transpiler_program, place, filter_int8=True)
        test_prog = inference_transpiler_program

    if with_memory_optimization:
        fluid.memory_optimize(test_prog)

    exe = fluid.Executor(place)
    exe.run(startup_prog)

    fluid.io.load_inference_model(model_path,
            exe, 
            model_filename="__model__",
            params_filename="__params__")

    """
    visible_device = os.getenv('CUDA_VISIBLE_DEVICES')
    if visible_device:
        device_num = len(visible_device.split(','))
    else:
        device_num = subprocess.check_output(
            ['nvidia-smi', '-L']).decode().count('\n')
    """
    device_num = 1

    test_batch_size = 16
    if not args.enable_ce:
        test_reader = paddle.batch(reader.val(), batch_size=test_batch_size)
    else:
        # use flowers dataset for CE and set use_xmap False to avoid disorder data
        # but it is time consuming. For faster speed, need another dataset.
        import random
        random.seed(0)
        np.random.seed(0)
        test_reader = paddle.batch(
            flowers.test(use_xmap=False), batch_size=test_batch_size)

    test_py_reader.decorate_paddle_reader(test_reader)

    test_fetch_list = [test_cost.name, test_acc1.name, test_acc5.name]

    for pass_id in range(args.num_epochs):

        test_py_reader.start()
        test_info = [[], [], []]

        test_batch_id = 0
        try:
            while True:
                t1 = time.time()
                loss, acc1, acc5 = exe.run(program=test_prog,
                                           fetch_list=test_fetch_list)
                t2 = time.time()
                period = t2 - t1
                loss = np.mean(loss)
                acc1 = np.mean(acc1)
                acc5 = np.mean(acc5)
                test_info[0].append(loss)
                test_info[1].append(acc1)
                test_info[2].append(acc5)
                if test_batch_id % 10 == 0:
                    print("Pass {0},testbatch {1},loss {2}, \
                        acc1 {3},acc5 {4},time {5}"
                          .format(pass_id, test_batch_id, loss, acc1, acc5,
                                  "%2.2f sec" % period))
                    sys.stdout.flush()
                test_batch_id += 1
        except fluid.core.EOFException:
            test_py_reader.reset()

        test_loss = np.array(test_info[0]).mean()
        test_acc1 = np.array(test_info[1]).mean()
        test_acc5 = np.array(test_info[2]).mean()

        print("test_loss {0}, test_acc1 {1}, test_acc5 {2}".format(
            test_loss, test_acc1, test_acc5))
        sys.stdout.flush()

def main():
    args = parser.parse_args()
    set_models(args.model_category)
    print_arguments(args)

    model_path = os.path.join(args.model_save_dir + '/' + args.model)

    if args.place == "cuda":
        place = fluid.CUDAPlace(0)
    elif args.place == "xsim":
        place = fluid.XSIMPlace()
    elif args.place == "xpu":
        place = fluid.XPUPlace()
    else:
        print("Unsurpported place!")
        exit()

    if (args.run_mode == "train"):
        train(args, model_path, place)
    elif (args.run_mode == "infer" or
            args.run_mode == "fused_infer"):
        infer(args, model_path, place)


if __name__ == '__main__':
    main()
