from monai.transforms import RandZoomd, RandRotated, RandGaussianNoised, Resized, RepeatChanneld

from mmpretrain.registry import TRANSFORMS


@TRANSFORMS.register_module()
class MonaiResized(Resized):
    pass


@TRANSFORMS.register_module()
class MonaiRandZoomd(RandZoomd):
    pass


@TRANSFORMS.register_module()
class MonaiRandRotated(RandRotated):
    pass


@TRANSFORMS.register_module()
class MonaiRandGaussianNoised(RandGaussianNoised):
    pass


@TRANSFORMS.register_module()
class MonaiRepeatChanneld(RepeatChanneld):
    pass
