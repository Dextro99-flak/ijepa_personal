import torchvision.transforms as T
def make_transforms(*args, **kwargs):
# Only spatial transforms are safe for 224-channel arrays
return T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip()
])
