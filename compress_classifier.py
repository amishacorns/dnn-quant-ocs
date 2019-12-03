#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""This is an example application for compressing image classification models.

The application borrows its main flow code from torchvision's ImageNet classification
training sample application (https://github.com/pytorch/examples/tree/master/imagenet).
We tried to keep it similar, in order to make it familiar and easy to understand.

Integrating compression is very simple: simply add invocations of the appropriate
compression_scheduler callbacks, for each stage in the training.  The training skeleton
looks like the pseudo code below.  The boiler-plate Pytorch classification training
is speckled with invocations of CompressionScheduler.

For each epoch:
    compression_scheduler.on_epoch_begin(epoch)
    train()
    validate()
    save_checkpoint()
    compression_scheduler.on_epoch_end(epoch)

train():
    For each training step:
        compression_scheduler.on_minibatch_begin(epoch)
        output = model(input)
        loss = criterion(output, target)
        compression_scheduler.before_backward_pass(epoch)
        loss.backward()
        optimizer.step()
        compression_scheduler.on_minibatch_end(epoch)


This exmple application can be used with torchvision's ImageNet image classification
models, or with the provided sample models:

- ResNet for CIFAR: https://github.com/junyuseu/pytorch-cifar-models
- MobileNet for ImageNet: https://github.com/marvis/pytorch-mobilenet
"""

import argparse
import time
import os
import sys
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchnet.meter as tnt
script_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(script_dir, '..'))
try:
    import distiller
except ImportError:
    sys.path.append(module_path)
    import distiller
import apputils
import distiller.quantization as quantization
from models import ALL_MODEL_NAMES, create_model

# Logger handle
msglogger = None


def float_range(val_str):
    val = float(val_str)
    if val < 0 or val >= 1:
        raise argparse.ArgumentTypeError('Must be >= 0 and < 1 (received {0})'.format(val_str))
    return val


def create_parser():
    parser = argparse.ArgumentParser(description='Distiller image classification model compression')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=ALL_MODEL_NAMES,
                        help='model architecture: ' +
                        ' | '.join(ALL_MODEL_NAMES) +
                        ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--act-stats', dest='activation_stats', action='store_true', default=False,
                        help='collect activation statistics (WARNING: this slows down training)')
    parser.add_argument('--param-hist', dest='log_params_histograms', action='store_true', default=False,
                        help='log the paramter tensors histograms to file (WARNING: this can use significant disk space)')
    SUMMARY_CHOICES = ['sparsity', 'compute', 'model', 'modules', 'png', 'png_w_params', 'onnx']
    parser.add_argument('--summary', type=str, choices=SUMMARY_CHOICES,
                        help='print a summary of the model, and exit - options: ' +
                        ' | '.join(SUMMARY_CHOICES))
    parser.add_argument('--compress', dest='compress', type=str, nargs='?', action='store',
                        help='configuration file for pruning the model (default is to use hard-coded schedule)')
    parser.add_argument('--sense', dest='sensitivity', choices=['element', 'filter', 'channel'],
                        help='test the sensitivity of layers to pruning')
    parser.add_argument('--extras', default=None, type=str,
                        help='file with extra configuration information')
    parser.add_argument('--deterministic', '--det', action='store_true',
                        help='Ensure deterministic execution for re-producible results.')
    parser.add_argument('--quantize-method', default=None, type=str,
                        choices=[None, "linear", "ocs"],
                        help='Apply quantization to model before evaluation')
    parser.add_argument('--act-bits', default=8, type=int,
                        help='Number of bits in quantized activations')
    parser.add_argument('--weight-bits', default=8, type=int,
                        help='Number of bits in quantized weights')

    parser.add_argument('--weight-expand-ratio', default=0.0, type=float_range,
                        help='The weight expand ratio in OCSQuantizer')
    parser.add_argument('--act-expand-ratio', default=0.0, type=float_range,
                        help='The act expand ratio in OCSQuantizer')
    parser.add_argument('--weight-clip-threshold', default=1.0, type=float,
                        help='The weight clip threshold in OCSQuantizer. Use 0 for mmse clipping')
    parser.add_argument('--act-clip-threshold', default=1.0, type=float,
                        help='The act clip threshold in OCSQuantizer. Use 0 for mmse clipping')
    parser.add_argument('--profile-batches', default=1, type=int,
                        help='The number of train batches to profile for OCSQuantizer')

    parser.add_argument('--gpus', metavar='DEV_ID', default=None,
                        help='Comma-separated list of GPU device IDs to be used (default is to use all available devices)')
    parser.add_argument('--name', '-n', metavar='NAME', default=None, help='Experiment name')
    parser.add_argument('--out-dir', '-o', dest='output_dir', default='logs', help='Path to dump logs and checkpoints')
    parser.add_argument('--validation-size', '--vs', type=float_range, default=0.0,
                        help='Portion of training dataset to set aside for validation')
    parser.add_argument('--adc', dest='ADC', action='store_true', help='temp HACK')
    parser.add_argument('--adc-params', dest='ADC_params', default=None, help='temp HACK')
    parser.add_argument('--confusion', dest='display_confusion', default=False, action='store_true',
                        help='Display the confusion matrix')
    parser.add_argument('--earlyexit_lossweights', type=float, nargs='*', dest='earlyexit_lossweights', default=None,
                        help='List of loss weights for early exits (e.g. --lossweights 0.1 0.3)')
    parser.add_argument('--earlyexit_thresholds', type=float, nargs='*', dest='earlyexit_thresholds', default=None,
                        help='List of EarlyExit thresholds (e.g. --earlyexit 1.2 0.9)')

    distiller.knowledge_distillation.add_distillation_args(parser, ALL_MODEL_NAMES, True)
    return parser

def main_func(args=None):
    global msglogger
    parser = create_parser()
    # parse the args if they aren't provided
    if not args:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args=args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    msglogger = apputils.config_pylogger(os.path.join(script_dir, 'logging.conf'), args.name, args.output_dir)

    msglogger.debug("Distiller: %s", distiller.__version__)

    if args.gpus is not None:
        try:
            args.gpus = [int(s) for s in args.gpus.split(',')]
        except ValueError:
            msglogger.error('ERROR: Argument --gpus must be a comma-separated list of integers only')
            exit(1)
        available_gpus = torch.cuda.device_count()
        for dev_id in args.gpus:
            if dev_id >= available_gpus:
                msglogger.error('ERROR: GPU device ID {0} requested, but only {1} devices available'
                                .format(dev_id, available_gpus))
                exit(1)
        # Set default device in case the first one on the list != 0
        torch.cuda.set_device(args.gpus[0])

    # Infer the dataset from the model name
    args.dataset = 'cifar10' if 'cifar' in args.arch else 'imagenet'
    args.num_classes = 10 if args.dataset == 'cifar10' else 1000

    # Create the model
    model = create_model(args.pretrained, args.dataset, args.arch, device_ids=args.gpus)

    # We can optionally resume from a checkpoint
    if args.resume:
        model, compression_scheduler, start_epoch = apputils.load_checkpoint(
            model, chkpt_file=args.resume)

    # Load the datasets: the dataset to load is inferred from the model name passed
    # in args.arch.  The default dataset is ImageNet, but if args.arch contains the
    # substring "_cifar", then cifar10 is used.
    #
    # RZ: inception models use a 299x299 center crop
    crop_size = 299 if 'inception' in args.arch.lower() else 224
    train_loader, val_loader, test_loader, _ = apputils.load_data(
        args.dataset, os.path.expanduser(args.data), args.batch_size,
        args.workers, args.validation_size, args.deterministic, crop_size=crop_size)
    msglogger.info('Dataset sizes:\n\ttraining=%d\n\tvalidation=%d\n\ttest=%d',
                   len(train_loader.sampler), len(val_loader.sampler), len(test_loader.sampler))

    if args.evaluate:
        # quantize_model(model, train_loader, args)
        classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, 5))

        total_samples = len(test_loader.sampler)
        batch_size = test_loader.batch_size
        msglogger.info('%d samples (%d per mini-batch)', total_samples, batch_size)

        # Switch to evaluation mode
        model.eval()

        for validation_step, (inputs, target) in enumerate(test_loader):
            with torch.no_grad():
                inputs, target = inputs.to('cuda'), target.to('cuda')
                # compute output from model
                output = model(inputs)

                # measure accuracy
                classerr.add(output.data, target)

        msglogger.info('==> Top1: %.3f    Top5: %.3f', classerr.value()[0], classerr.value()[1])

        return classerr.value(1)


def profile_for_quantization(data_loader, model, args):
    """Profile activations for quantization"""
    msglogger.info('--- profile for quantization ---')

    # Switch to evaluation mode
    model.eval()

    if args.profile_batches <= 0:
        return

    end = time.time()

    data_iter = iter(data_loader)
    for profile_batch in range(args.profile_batches):
        if profile_batch == 0:
            (inputs, target) = next(data_iter)
        else:
            (inputs_i, target_i) = next(data_iter)
            inputs = torch.cat([inputs, inputs_i], dim=0)
            target = torch.cat([target, target_i], dim=0)

    msglogger.info('--- profiling with %d images ---' % inputs.shape[0])

    with torch.no_grad():
        inputs, target = inputs.to('cuda'), target.to('cuda')
        # compute output from model
        output = model(inputs)

    # measure elapsed time
    msglogger.info('==> Profile runtime: %d' % (time.time() - end))


def quantize_model(model, train_loader, args):
    if args.quantize_method:
        if args.quantize_method == "linear":
            quantizer = quantization.SymmetricLinearQuantizer(model,
                args.act_bits, args.weight_bits)
        if args.quantize_method == "ocs":
            quantizer = quantization.OCSQuantizer(model,
                args.act_bits, args.weight_bits,
                weight_expand_ratio=args.weight_expand_ratio,
                weight_clip_threshold=args.weight_clip_threshold,
                act_expand_ratio=args.act_expand_ratio,
                act_clip_threshold=args.act_clip_threshold)
        else:
            msglogger.info('--- Quantizer not found ---')

        model.cpu()
        quantizer.prepare_model()
        model.cuda()

        if args.quantize_method == "ocs":
            # Profile the activation first for ocs
            quantization.ocs_set_profile_mode(True)
            profile_for_quantization(train_loader, model, args)
            quantization.ocs_set_profile_mode(False)

print(main_func())
