'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

'''
Usage:
python onnx_export.py  --arch zennet_gesture_model_size97k_flops3.71M_acc94.48_res96 --shape 96 --out_file out.onnx --gpu 0
'''
import sys, argparse
import torch
import ZenNet
from torch.autograd import Variable


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default=None,
                        help='model to be evaluated.')
    parser.add_argument('--shape', type=int,  nargs='+', default=[96, 96],
                        help='cifar10 or cifar100')
    parser.add_argument('--out_file', type=str, default="out.onnx",
                        help='out onnx file name.')
    parser.add_argument('--gpu', type=int, default=None,
                        help='input image shape.')
    opt, _ = parser.parse_known_args(sys.argv)

    
    # load model
    model = ZenNet.get_ZenNet(opt.arch, pretrained=True)

    # get input
    shape = opt.shape
    if len(shape) == 1:
        img_shape = (1, 3, shape[0], shape[0])
    elif len(shape) == 2:
        img_shape = (1, 3) + tuple(shape)
    elif len(shape) == 4:
        img_shape = tuple(shape)
    else:
        raise ValueError('invalid input shape')
    print("---->img_shape: ", img_shape)
    # img_shape = (1, 3, 96, 96) # give input_shape directly
    dummy_input = Variable(torch.randn(*img_shape, device='cpu'))

    # set output
    out_file = opt.out_file

    # use gpu or not
    if opt.gpu is not None:
        torch.cuda.set_device(opt.gpu)
        torch.backends.cudnn.benchmark = True
        model = model.cuda(opt.gpu)
        print('---->Using GPU {}.'.format(opt.gpu))
        dummy_input = dummy_input.cuda(opt.gpu, non_blocking=True)

    model.eval()

    # onnx conversion part
    print("---->begin to convert onnx")
    torch.onnx.export(model, dummy_input, out_file)
    print("---->onnx conversion complete")

