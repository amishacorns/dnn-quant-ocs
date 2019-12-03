import subprocess as sp
import uptune as ut
from compress_classifier import main_func

data_dir = '/work/zhang-x2/common/datasets/imagenet-pytorch/'
model = 'resnet50'
# MODEL_PATH = "OCS-CNN/logs/2019.10.21-194654/best.pth.tar"
out_dir = 'logs_' + model
# abits = ut.tune(8, (4, 8))
# wbits = ut.tune(8, (4, 8))
abits = 8
wbits = 8
# w_thresh = ut.tune(.80, (.60, 1.0))
# a_thresh = ut.tune(.80, (.60, 1.0))
w_thresh = 1.0
a_thresh = 1.0
# TODO skip quantizing the first and last layers for no

exp_name = 'tuning-%da%dw-%2.2ft_w%2.2ft_a' % (abits, wbits, w_thresh, a_thresh)
args = ["%s" % data_dir,
        "--arch=%s" % model,
        "--evaluate",
        # "--resume=%s" % MODEL_PATH,
        "--pretrained",
        "--act-bits=%d" % abits,
        "--weight-bits=%d" % wbits,
        "--quantize-method=%s" % "ocs",
        "--weight-clip-threshold=%5.5f" % w_thresh,
        "--act-clip-threshold=%5.5f" % a_thresh,
        "--profile-batches=4",
        "-b 128",
        "-j 1",
        "--gpu=0",  # arbitrary GPU since parallelization doesn't work
        "--vs=0",
        "--out-dir=%s" % out_dir,
        "--name=%s" % exp_name
        ]

print("Args:")
print(args)

val_acc = main_func(args)
print(val_acc)
ut.target(val_acc, objective='max')
