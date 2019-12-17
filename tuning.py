import uptune as ut
from tuning_evaluate import main_func

data_dir = '/work/zhang-x2/common/datasets/imagenet-pytorch/'
model = 'resnet18'
out_dir = 'logs_' + model
abits = 5
wbits = 5
# w_thresh = ut.tune(.80, (.60, 1.0))
# a_thresh = ut.tune(.80, (.60, 1.0))
w_thresh = 1.0
a_thresh = 1.0

exp_name = 'tuning-%da%dw-%2.2ft_w%2.2ft_a' % (abits, wbits, w_thresh, a_thresh)
args = ["%s" % data_dir,
        "--arch=%s" % model,
        "--act-bits=%d" % abits,
        "--weight-bits=%d" % wbits,
        "--quantize-method=%s" % "ocs",
        "--weight-clip-threshold=%5.5f" % w_thresh,
        "--act-clip-threshold=%5.5f" % a_thresh,
        "--vs=.0005",
        "-b 128",
        "-j 1",
        "--gpu=0",  # arbitrary GPU since parallelization doesn't work
        "--out-dir=%s" % out_dir,
        "--name=%s" % exp_name
        ]

print("Args:")
print(args)

val_acc = main_func(args)
print(val_acc)
ut.target(val_acc, objective='max')
