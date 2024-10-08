from torchvision import transforms
from .office_loader import OfficeDataset
import torch
from .domainnet_loader import DomainNetDataset

mean=[0.5, 0.5, 0.5]
std=[0.5, 0.5, 0.5]


def prepare_caltech(cfg):
    data_base_path = cfg.DATA.DATAPATH
    num_workers = cfg.DATA.NUM_WORKERS
    train_ratio = cfg.DATA.TRAIN_RATIO
    assert train_ratio <= 0.5

    transform_office = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop([224,224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
    ])

    transform_test = transforms.Compose([
            transforms.Resize([256, 256]),   
            transforms.CenterCrop([224,224]),         
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
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
    min_data_len = min(len(amazon_trainset), len(caltech_trainset), len(dslr_trainset), len(webcam_trainset))
    val_len = int(min_data_len * train_ratio)
    min_data_len = int(min_data_len * train_ratio)
    # print(f'val_len: {val_len}')
    # min_data_len = int(min_data_len * 0.5)

    amazon_valset = torch.utils.data.Subset(amazon_trainset, list(range(len(amazon_trainset)))[-val_len:]) 
    amazon_trainset = torch.utils.data.Subset(amazon_trainset, list(range(min_data_len)))
    # f.write(f'amazon_train: {len(amazon_trainset)} \n')
    # f.write(f'amazon_val: {len(amazon_valset)} \n')
    # f.write(f'amazon_test: {len(amazon_testset)} \n')
    caltech_valset = torch.utils.data.Subset(caltech_trainset, list(range(len(caltech_trainset)))[-val_len:]) 
    caltech_trainset = torch.utils.data.Subset(caltech_trainset, list(range(min_data_len)))
    # f.write(f'caltech_train: {len(caltech_trainset)} \n')
    # f.write(f'caltech_val: {len(caltech_valset)} \n')
    # f.write(f'caltech_test: {len(caltech_testset)} \n')

    dslr_valset = torch.utils.data.Subset(dslr_trainset, list(range(len(dslr_trainset)))[-val_len:]) 
    dslr_trainset = torch.utils.data.Subset(dslr_trainset, list(range(min_data_len)))
    # f.write(f'dslr_train: {len(dslr_trainset)} \n')
    # f.write(f'dslr_val: {len(dslr_valset)} \n')
    # f.write(f'dslr_test: {len(dslr_testset)} \n')

    webcam_valset = torch.utils.data.Subset(webcam_trainset, list(range(len(webcam_trainset)))[-val_len:]) 
    webcam_trainset = torch.utils.data.Subset(webcam_trainset, list(range(min_data_len)))
    # f.write(f'webcam_train: {len(webcam_trainset)} \n')
    # f.write(f'webcam_val: {len(webcam_valset)} \n')
    # f.write(f'webcam_test: {len(webcam_testset)} \n')
    # f.close()
    # print(len(amazon_valset)+len(caltech_valset)+len(dslr_valset)+len(webcam_valset)+len(amazon_trainset)+len(amazon_testset)+len(caltech_trainset)+len(caltech_testset)+len(dslr_trainset)+len(dslr_testset)+len(webcam_trainset)+len(webcam_testset))
    # exit()
    amazon_val_loader = torch.utils.data.DataLoader(amazon_valset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=num_workers)
    caltech_val_loader = torch.utils.data.DataLoader(caltech_valset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=num_workers)
    dslr_val_loader = torch.utils.data.DataLoader(dslr_valset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=num_workers)
    webcam_val_loader = torch.utils.data.DataLoader(webcam_valset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=num_workers)
    
    amazon_train_loader = torch.utils.data.DataLoader(amazon_trainset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=num_workers)
    amazon_test_loader = torch.utils.data.DataLoader(amazon_testset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=num_workers)

    caltech_train_loader = torch.utils.data.DataLoader(caltech_trainset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=num_workers)
    caltech_test_loader = torch.utils.data.DataLoader(caltech_testset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=num_workers)

    dslr_train_loader = torch.utils.data.DataLoader(dslr_trainset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=num_workers)
    dslr_test_loader = torch.utils.data.DataLoader(dslr_testset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=num_workers)

    webcam_train_loader = torch.utils.data.DataLoader(webcam_trainset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=num_workers)
    webcam_test_loader = torch.utils.data.DataLoader(webcam_testset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=num_workers)
    
    train_loaders = [amazon_train_loader, caltech_train_loader, dslr_train_loader, webcam_train_loader]
    val_loaders = [amazon_val_loader, caltech_val_loader, dslr_val_loader, webcam_val_loader]
    test_loaders = [amazon_test_loader, caltech_test_loader, dslr_test_loader, webcam_test_loader]

    cnt = 0
    site = ['amazon', 'caltech', 'dslr', 'webcam']
    i=0
    for a, b, c in zip(train_loaders, val_loaders, test_loaders):
        cnt += len(a.dataset) + len(b.dataset) + len(c.dataset)
        print(f'{site[i]} Train: {len(a.dataset)}')
        print(f'{site[i]} Val: {len(b.dataset)}')
        print(f'{site[i]} Test: {len(c.dataset)}')
        i+=1
    assert cnt == 2533
    
    return site, train_loaders, val_loaders, test_loaders



def prepare_domainnet(cfg):
    data_base_path = cfg.DATA.DATAPATH
    transform_train = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop([224,224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
    ])

    transform_test = transforms.Compose([
            transforms.Resize([256, 256]),   
            transforms.CenterCrop([224,224]),         
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
    ])
    train_ratio = cfg.DATA.TRAIN_RATIO
    assert train_ratio <= 0.5

    min_data_len = 5e8
    tmp = list()
    for site in ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']:
        tmp.append(DomainNetDataset(data_base_path, site, transform=transform_train, train=True))
        if min_data_len > len(tmp[-1]):
            min_data_len = len(tmp[-1])

    val_len = int(min_data_len * train_ratio)
    min_data_len = int(min_data_len * train_ratio)

    train_loaders = list()
    val_loaders = list()
    test_loaders = list()

    for site in ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']:
        trainset = DomainNetDataset(data_base_path, site, transform=transform_train, train=True)
        testset = DomainNetDataset(data_base_path, site, transform=transform_test, train=False)
        cnt = len(trainset) + len(testset)
        
        # valset = torch.utils.data.Subset(trainset, list(range(len(trainset)))[-val_len:])
        # trainset = torch.utils.data.Subset(trainset, list(range(len(trainset)))[:-val_len])
        valset = torch.utils.data.Subset(trainset, list(range(len(trainset)))[-val_len:])
        trainset = torch.utils.data.Subset(trainset, list(range(min_data_len)))

        train_loaders.append(torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, pin_memory=True, num_workers=cfg.DATA.NUM_WORKERS))
        val_loaders.append(torch.utils.data.DataLoader(valset, batch_size=32, shuffle=False, pin_memory=True, num_workers=cfg.DATA.NUM_WORKERS))
        test_loaders.append(torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, pin_memory=True, num_workers=cfg.DATA.NUM_WORKERS))
        # assert cnt == len(train_loaders[-1].dataset) + len(val_loaders[-1].dataset) + len(test_loaders[-1].dataset)
        print(f'{site} Train: {len(train_loaders[-1].dataset)}')
        print(f'{site} Val: {len(val_loaders[-1].dataset)}')
        print(f'{site} Test: {len(test_loaders[-1].dataset)}')
    
    site = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    
    return site, train_loaders, val_loaders, test_loaders
