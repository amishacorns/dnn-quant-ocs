import subprocess as sp
import uptune as ut
from compress_classifier import main_func

data_dir = 'OCS-CNN/data.cifar'
model = 'simplenet_cifar'
MODEL_PATH = "OCS-CNN/logs/2019.10.21-194654/best.pth.tar"
out_dir = 'logs_' + model
# abits = ut.tune(8, (4, 8))
# wbits = ut.tune(8, (4, 8))
abits = 8
wbits = 8
# thresh = ut.tune(.80, (.50, 1.0))
thresh = .5
# TODO skip quantizing the first and last layers for no

exp_name = 'tuning-%da%dw-t%s' % (abits, wbits, thresh)
args = ["%s" % data_dir,
        "--arch=%s" % model,
        "--evaluate",
        "--resume=%s" % MODEL_PATH,
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
