import torchvision.transforms as T
def make_transforms(*args, **kwargs):
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip()
    ])
