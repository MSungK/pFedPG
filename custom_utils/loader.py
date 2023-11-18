from torchvision import transforms
from .office_loader import OfficeDataset
import torch


def prepare_data(cfg, use_val=True):
    data_base_path = cfg.DATA.DATAPATH
    transform_office = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.CenterCrop([224,224]),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([256, 256]),   
            transforms.CenterCrop([224,224]),         
            transforms.ToTensor(),
    ])
    
    # amazon
    amazon_trainset = OfficeDataset(data_base_path, 'amazon', transform=transform_office)
    amazon_testset = OfficeDataset(data_base_path, 'amazon', transform=transform_test, train=False)
    # caltech
    caltech_trainset = OfficeDataset(data_base_path, 'caltech', transform=transform_office)
    caltech_testset = OfficeDataset(data_base_path, 'caltech', transform=transform_test, train=False)
    # dslr
    dslr_trainset = OfficeDataset(data_base_path, 'dslr', transform=transform_office)
    dslr_testset = OfficeDataset(data_base_path, 'dslr', transform=transform_test, train=False)
    # webcam
    webcam_trainset = OfficeDataset(data_base_path, 'webcam', transform=transform_office)
    webcam_testset = OfficeDataset(data_base_path, 'webcam', transform=transform_test, train=False)

    # f = open('OfficeCaltech10.txt', 'w')

    # f.write(f'total: {len(amazon_trainset)+len(amazon_testset)+len(caltech_trainset)+len(caltech_testset)+len(dslr_trainset)+len(dslr_testset)+len(webcam_trainset)+len(webcam_testset)} \n')
    if use_val:
        min_data_len = min(len(amazon_trainset), len(caltech_trainset), len(dslr_trainset), len(webcam_trainset))
        val_len = int(min_data_len * 0.3)
        # print(f'val_len: {val_len}')
        # min_data_len = int(min_data_len * 0.5)

        amazon_valset = torch.utils.data.Subset(amazon_trainset, list(range(len(amazon_trainset)))[-val_len:]) 
        amazon_trainset = torch.utils.data.Subset(amazon_trainset, list(range(len(amazon_trainset)-val_len)))
        # f.write(f'amazon_train: {len(amazon_trainset)} \n')
        # f.write(f'amazon_val: {len(amazon_valset)} \n')
        # f.write(f'amazon_test: {len(amazon_testset)} \n')
        caltech_valset = torch.utils.data.Subset(caltech_trainset, list(range(len(caltech_trainset)))[-val_len:]) 
        caltech_trainset = torch.utils.data.Subset(caltech_trainset, list(range(len(caltech_trainset)-val_len)))
        # f.write(f'caltech_train: {len(caltech_trainset)} \n')
        # f.write(f'caltech_val: {len(caltech_valset)} \n')
        # f.write(f'caltech_test: {len(caltech_testset)} \n')

        dslr_valset = torch.utils.data.Subset(dslr_trainset, list(range(len(dslr_trainset)))[-val_len:]) 
        dslr_trainset = torch.utils.data.Subset(dslr_trainset, list(range(len(dslr_trainset)-val_len)))
        # f.write(f'dslr_train: {len(dslr_trainset)} \n')
        # f.write(f'dslr_val: {len(dslr_valset)} \n')
        # f.write(f'dslr_test: {len(dslr_testset)} \n')

        webcam_valset = torch.utils.data.Subset(webcam_trainset, list(range(len(webcam_trainset)))[-val_len:]) 
        webcam_trainset = torch.utils.data.Subset(webcam_trainset, list(range(len(webcam_trainset)-val_len)))
        # f.write(f'webcam_train: {len(webcam_trainset)} \n')
        # f.write(f'webcam_val: {len(webcam_valset)} \n')
        # f.write(f'webcam_test: {len(webcam_testset)} \n')
        # f.close()
        # print(len(amazon_valset)+len(caltech_valset)+len(dslr_valset)+len(webcam_valset)+len(amazon_trainset)+len(amazon_testset)+len(caltech_trainset)+len(caltech_testset)+len(dslr_trainset)+len(dslr_testset)+len(webcam_trainset)+len(webcam_testset))
        # exit()
        amazon_val_loader = torch.utils.data.DataLoader(amazon_valset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=False)
        caltech_val_loader = torch.utils.data.DataLoader(caltech_valset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=False)
        dslr_val_loader = torch.utils.data.DataLoader(dslr_valset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=False)
        webcam_val_loader = torch.utils.data.DataLoader(webcam_valset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=False)
        val_loaders = [amazon_val_loader, caltech_val_loader, dslr_val_loader, webcam_val_loader]
    
    amazon_train_loader = torch.utils.data.DataLoader(amazon_trainset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True)
    amazon_test_loader = torch.utils.data.DataLoader(amazon_testset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=False)

    caltech_train_loader = torch.utils.data.DataLoader(caltech_trainset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True)
    caltech_test_loader = torch.utils.data.DataLoader(caltech_testset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=False)

    dslr_train_loader = torch.utils.data.DataLoader(dslr_trainset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True)
    dslr_test_loader = torch.utils.data.DataLoader(dslr_testset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=False)

    webcam_train_loader = torch.utils.data.DataLoader(webcam_trainset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True)
    webcam_test_loader = torch.utils.data.DataLoader(webcam_testset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=False)
    
    train_loaders = [amazon_train_loader, caltech_train_loader, dslr_train_loader, webcam_train_loader]
    test_loaders = [amazon_test_loader, caltech_test_loader, dslr_test_loader, webcam_test_loader]
    
    if use_val:
        return train_loaders, val_loaders, test_loaders
    return train_loaders, test_loaders