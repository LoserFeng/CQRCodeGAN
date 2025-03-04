
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'pix2pix':
        assert(opt.dataset_mode == 'pix2pix')
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'pair':
        # assert(opt.dataset_mode == 'pair')
        # from .pair_model import PairModel
        from .Unet_L1 import PairModel
        model = PairModel()
    elif opt.model == 'single':
        # assert(opt.dataset_mode == 'unaligned')
        from .single_model import SingleModel
        model = SingleModel()
    elif opt.model == 'temp':
        # assert(opt.dataset_mode == 'unaligned')
        from .temp_model import TempModel
        model = TempModel()
    elif opt.model == 'UNIT':
        assert(opt.dataset_mode == 'unaligned')
        from .unit_model import UNITModel
        model = UNITModel()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()


    elif opt.model=='CQRCodeGAN':
        assert(opt.dataset_mode=='unaligned')
        from .qrcode_model import CQRCodeGANModel
        model=CQRCodeGANModel()
    elif opt.model=='CQRCodePatchGAN':
        assert(opt.dataset_mode=='unaligned_patch')
        from .qrcode_patch_model import CQRCodePatchGANModel
        model=CQRCodePatchGANModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
