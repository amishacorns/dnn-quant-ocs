import subprocess as sp
import uptune as ut
from compress_classifier import main_func

data_dir = '/work/zhang-x2/common/datasets/imagenet-pytorch/'
model = 'resnet18'
out_dir = 'logs_' + model
# abits = ut.tune(8, (4, 8))
# wbits = ut.tune(8, (4, 8))
abits = 8
wbits = 8
# thresh = ut.tune(.80, (.50, 1.0))
thresh = 1.0
# TODO skip quantizing the first and last layers for no

exp_name = 'tuning-%da%dw-t%s' % (abits, wbits, thresh)
args = ["%s" % data_dir,
        "--arch=%s" % model,
        "--evaluate",
        "--pretrained",
        "--act-bits=%d" % abits,
        "--weight-bits=%d" % wbits,
        "--quantize-method=%s" % "ocs",
        "--weight-clip-threshold=%d" % thresh,
        "--act-clip-threshold=1.0",
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
ut.target(val_acc, objective='max')
