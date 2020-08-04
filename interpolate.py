#!/usr/bin/env python3
import argparse
import os
import os.path
import ctypes
from shutil import rmtree, move
from PIL import Image
import torch
import torchvision.transforms as transforms
import model
import dataloader
import platform
from tqdm import tqdm

# For parsing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--ffmpeg_dir", type=str, default="", help='path to ffmpeg.exe')
parser.add_argument("--start", type=str, required=True, help="First image")
parser.add_argument("--end", type=str, required=True, help="Last image")
parser.add_argument("--checkpoint", type=str, required=True, help='path of checkpoint for pretrained model')
parser.add_argument("--sf", type=int, required=True, help='specify the slomo factor N. This will increase the frames by Nx. Example sf=2 ==> 2x frames')
parser.add_argument("--batch_size", type=int, default=1, help='Specify batch size for faster conversion. This will depend on your cpu/gpu memory. Default: 1')
parser.add_argument("--output_dir", type=str, default="output", help='Specify output directory. Default: output/')
args = parser.parse_args()

def check():
    """
    Checks the validity of commandline arguments.

    Parameters
    ----------
        None

    Returns
    -------
        error : string
            Error message if error occurs otherwise blank string.
    """


    error = ""
    if (args.sf < 2):
        error = "Error: --sf/slomo factor has to be atleast 2"
    if (args.batch_size < 1):
        error = "Error: --batch_size has to be atleast 1"
    return error

def extract_frames(video, outDir):
    """
    Converts the `video` to images.

    Parameters
    ----------
        video : string
            full path to the video file.
        outDir : string
            path to directory to output the extracted images.

    Returns
    -------
        error : string
            Error message if error occurs otherwise blank string.
    """


    error = ""
    print('{} -i {} -vsync 0 {}/%06d.png'.format(os.path.join(args.ffmpeg_dir, "ffmpeg"), video, outDir))
    retn = os.system('{} -i "{}" -vsync 0 {}/%06d.png'.format(os.path.join(args.ffmpeg_dir, "ffmpeg"), video, outDir))
    if retn:
        error = "Error converting file:{}. Exiting.".format(video)
    return error

def create_video(dir):
    error = ""
    print('{} -r {} -i {}/%d.png -vcodec ffvhuff {}'.format(os.path.join(args.ffmpeg_dir, "ffmpeg"), args.fps, dir, args.output))
    retn = os.system('{} -r {} -i {}/%d.png -vcodec ffvhuff "{}"'.format(os.path.join(args.ffmpeg_dir, "ffmpeg"), args.fps, dir, args.output))
    if retn:
        error = "Error creating output video. Exiting."
    return error


class Interpolator(object):
    def __init__(self, checkpoint, frame0, frame1, batch_size=1):
        # Initialize transforms
        self.checkpoint = checkpoint
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        mean = [0.429, 0.431, 0.397]
        std  = [1, 1, 1]
        normalize = transforms.Normalize(mean=mean,
                                        std=std)

        negmean = [x * -1 for x in mean]
        revNormalize = transforms.Normalize(mean=negmean, std=std)

        # Temporary fix for issue #7 https://github.com/avinashpaliwal/Super-SloMo/issues/7 -
        # - Removed per channel mean subtraction for CPU.
        if (self.device == "cpu"):
            self.transform = transforms.Compose([transforms.ToTensor()])
            self.TP = transforms.Compose([transforms.ToPILImage()])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(), normalize])
            self.TP = transforms.Compose([revNormalize, transforms.ToPILImage()])

        # Load data
        self.videoFrames = dataloader.Images(frame0=frame0, frame1=frame1, transform=self.transform)
        self.videoFramesloader = torch.utils.data.DataLoader(self.videoFrames, batch_size=self.batch_size, shuffle=False)

        # Initialize model
        self.flowComp = model.UNet(6, 4)
        self.flowComp.to(self.device)
        for param in self.flowComp.parameters():
            param.requires_grad = False
        self.ArbTimeFlowIntrp = model.UNet(20, 5)
        self.ArbTimeFlowIntrp.to(self.device)
        for param in self.ArbTimeFlowIntrp.parameters():
            param.requires_grad = False

        self.flowBackWarp = model.backWarp(self.videoFrames.dim[0], self.videoFrames.dim[1], self.device)
        self.flowBackWarp = self.flowBackWarp.to(self.device)

        dict1 = torch.load(self.checkpoint, map_location='cpu')
        self.ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
        self.flowComp.load_state_dict(dict1['state_dictFC'])

    def interpolate(self, frame0, frame1, sf, outputPath=None):
        # Interpolate frames
        self.videoFrames = dataloader.Images(frame0=frame0, frame1=frame1, transform=self.transform)

        frameCounter = 1

        with torch.no_grad():
            for _, (frame0, frame1) in enumerate(self.videoFramesloader, 0):
                I0 = frame0.to(self.device)
                I1 = frame1.to(self.device)

                flowOut = self.flowComp(torch.cat((I0, I1), dim=1))
                F_0_1 = flowOut[:,:2,:,:]
                F_1_0 = flowOut[:,2:,:,:]

                # Save reference frames in output folder
                for batchIndex in range(self.batch_size):
                    img = (self.TP(frame0[batchIndex].detach())).resize(self.videoFrames.origDim, Image.BILINEAR)
                    if outputPath is None:
                        yield img
                    else:
                        img.save(os.path.join(outputPath, "{:08d}.png".format((frameCounter + sf * batchIndex))))
                frameCounter += 1

                # Generate intermediate frames
                for intermediateIndex in range(1, sf):
                    t = float(intermediateIndex) / sf
                    temp = -t * (1 - t)
                    fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

                    F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
                    F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

                    g_I0_F_t_0 = self.flowBackWarp(I0, F_t_0)
                    g_I1_F_t_1 = self.flowBackWarp(I1, F_t_1)

                    intrpOut = self.ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))

                    F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
                    F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
                    V_t_0   = torch.sigmoid(intrpOut[:, 4:5, :, :])
                    V_t_1   = 1 - V_t_0

                    g_I0_F_t_0_f = self.flowBackWarp(I0, F_t_0_f)
                    g_I1_F_t_1_f = self.flowBackWarp(I1, F_t_1_f)

                    wCoeff = [1 - t, t]

                    Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

                    # Save intermediate frame
                    for batchIndex in range(self.batch_size):
                        img = (self.TP(Ft_p[batchIndex].cpu().detach())).resize(self.videoFrames.origDim, Image.BILINEAR)
                        if outputPath is None:
                            yield img
                        else:
                            img.save(os.path.join(outputPath, "{:08d}.png".format((frameCounter + sf * batchIndex))))
                    frameCounter += 1

                # Set counter accounting for batching of frames
                frameCounter += sf * (self.batch_size - 1)


def main():
    # Check if arguments are okay
    error = check()
    if error:
        print(error)
        exit(1)
    
    # Create output folder
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    i = Interpolator(args.checkpoint, args.start, args.end, batch_size=args.batch_size)
    for t in i.interpolate(args.start, args.end, args.sf, outputPath=args.output_dir):
        print(t)
 

    exit(0)

main()
