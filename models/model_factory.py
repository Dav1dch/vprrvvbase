# Author: Jacek Komorowski
# Warsaw University of Technology

from models.minkloc import MinkLoc
from models.netvlad import NetVLADVGG16
from models.patchnetvlad import PatchNetVLADVGG16
from models.multiresnetvlad import MultiResNetVLADVGG16
from models.minkloc_multimodal import MinkLocMultimodal, ResnetFPN
from misc.utils import MinkLocParams


def model_factory(params: MinkLocParams):
    in_channels = 1

    # MinkLocMultimodal is our baseline MinkLoc++ model producing 256 dimensional descriptor where
    # each modality produces 128 dimensional descriptor
    # MinkLocRGB and MinkLoc3D are single-modality versions producing 256 dimensional descriptor
    if params.model_params.model == "MinkLocRGB":
        image_fe_size = 256
        image_fe = ResnetFPN(
            out_channels=image_fe_size,
            lateral_dim=image_fe_size,
            fh_num_bottom_up=4,
            fh_num_top_down=0,
        )
        model = MinkLocMultimodal(
            None, 0, image_fe, image_fe_size, output_dim=image_fe_size
        )
    elif params.model_params.model == "netvlad":
        image_fe_size = 16384
        image_fe = NetVLADVGG16()
        model = MinkLocMultimodal(
            None, 0, image_fe, image_fe_size, output_dim=image_fe_size
        )
    elif params.model_params.model == "patchnetvlad":
        image_fe_size = 16384
        image_fe = PatchNetVLADVGG16()
        model = MinkLocMultimodal(
            None, 0, image_fe, image_fe_size, output_dim=image_fe_size
        )
    elif params.model_params.model == "multiresnetvlad":
        image_fe_size = 16384
        image_fe = MultiResNetVLADVGG16()
        model = MinkLocMultimodal(
            None, 0, image_fe, image_fe_size, output_dim=image_fe_size
        )
    else:
        raise NotImplementedError(
            "Model not implemented: {}".format(params.model_params.model)
        )

    return model
