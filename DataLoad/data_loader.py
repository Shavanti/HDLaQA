import torch
import torchvision
import DataLoad.data_folders as folders


def datas(dataset, path, dtype, img_indx, patch_size, patch_num, batch_size, istrain=True):

    istrain = istrain

    if dataset == 'live':   # The smallest size is 438
        # Train transforms
        if istrain:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])
        # Test transforms
        else:
            transforms = torchvision.transforms.Compose([
                # torchvision.transforms.CenterCrop(400),
                torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])

    elif dataset == 'livec':
        if istrain:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])
        # Test transforms
        else:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])


    if dataset == 'live':
        data = folders.LIVEFolder(
            root=path, dis_type=dtype, index=img_indx, transform=transforms, patch_num=patch_num,
            batch_size=batch_size, istrain=istrain)
 
    elif dataset == 'livec':
        data = folders.LIVEChallengeFolder(root=path, index=img_indx, transform=transforms, patch_num=patch_num,
                                           batch_size=batch_size, istrain=istrain)
    return data

