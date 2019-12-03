import sys, os, re
import subprocess as sp

DATA_DIR = '../data.cifar'
MODEL_PATH = "logs/2019.10.21-194654/best.pth.tar"
MODEL = 'simplenet_cifar'
WBITS = [16,8,6,4, 3]
ABITS = 16

if __name__ == "__main__":
    out_dir = 'logs_' + MODEL
    for wbits in WBITS:

        exp_name = 'ocs-%da%dw-r%s' % (ABITS, wbits, r)

        args = [ "%s" % DATA_DIR,
                 "--arch=%s" % MODEL,
                 "--evaluate",
                 "--act-bits=%d" % ABITS,
                 "--resume=%s" % MODEL_PATH,
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

