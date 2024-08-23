# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import vhr_birdpose.pose_resnet
# import vhr_birdpose.pose_hrnet
# import vhr_birdpose.pose_vhr

from .pose_vhr import VHRBirdPose
from .pose_resnet import PoseResNet
from .pose_hrnet import PoseHighResolutionNet

__all__ = [
    'VHRBirdPose', 'PoseResNet', 'PoseHighResolutionNet'
]
