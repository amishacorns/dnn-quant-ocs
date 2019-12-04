import sys
import subprocess as sp

DATA_DIR = '/work/zhang-x2/common/datasets/imagenet-pytorch/'
MODELS = ['resnet50']
RATIOS = [1.0, 0]
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

        exp_name = 'clip-%da%dw-r%s' % (ABITS, wbits, r)

        args = [ "%s" % DATA_DIR,
                 "--arch=%s" % model,
                 "--evaluate",
                 "--pretrained",
                 "--act-bits=%d" % ABITS,
                 "--weight-bits=%d" % wbits,
                 "--quantize-method=%s" % "ocs",
                 "--weight-expand-ratio=0.0",
                 "--weight-clip-threshold=%5.3f" % r,
                 "--act-clip-threshold=1.0",
                 "--profile-batches=4",
                 "-b 128",
                 "--gpus=0",
                 "-j 1",
                 "--vs=0",
                 "--out-dir=%s" % out_dir,
                 "--name=%s" % exp_name]

        print("Args:")
        print(args)

        sp.call(["python", "compress_classifier.py"] + args)

