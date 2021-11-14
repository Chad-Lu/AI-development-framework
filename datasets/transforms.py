from torchvision import transforms
import PIL

def build_transform(cfg):

    transform = transforms.Compose([ transforms.Resize((224,224)),
                                     transforms.ColorJitter(brightness= 1.3, saturation=0.5, contrast=0.5),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.RandomVerticalFlip(p=0.5),
                                     transforms.RandomRotation(15, resample=PIL.Image.BILINEAR),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

    return transform