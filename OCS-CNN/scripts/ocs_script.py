import sys, os, re
import subprocess as sp

DATA_DIR = '/work/zhang-x2/common/datasets/imagenet-pytorch/'
MODELS = ['resnet18', 'densenet121', 'inception_v3']
RATIOS = [0, 0.01, 0.02, 0.05]
WBITS = [8,7,6,5]
ABITS = 8

if __name__ == "__main__":
  if DATA_DIR is None:
    print('Add the ImageNet data dir to the script.')
    sys.exit(0)

  for model in MODELS:
    out_dir = 'logs_' + model

    for r in RATIOS:
      for wbits in WBITS:

        exp_name = 'ocs-%da%dw-r%s' % (ABITS, wbits, r)

        args = [ "%s" % DATA_DIR,
                 "--arch=%s" % model,
                 "--evaluate",
                 "--pretrained",
                 "--act-bits=%d" % ABITS,
                 "--weight-bits=%d" % wbits,
                 "--quantize-method=%s" % "ocs",
                 "--weight-expand-ratio=%5.3f" % r,
                 "--weight-clip-threshold=1.0",
                 "--act-clip-threshold=1.0",
                 "--profile-batches=4",
                 "-b 128",
                 "-j 1",
                 "--vs=0",
                 "--gpu=0",
                 "--out-dir=%s" % out_dir,
                 "--name=%s" % exp_name]

        print("Args:")
        print(args)

        sp.call(["python", "compress_classifier.py"] + args)

