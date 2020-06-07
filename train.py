import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as T

from vae import VAE
from test_dataset import TestDataset
from validate import validate



def train(model, train_dataloader, optimizer, epoch, device, recon_dir):
    # toogle model to train mode
    model.train()
    train_loss = 0
    l = len(train_dataloader)
    for batch_idx, batch in enumerate(train_dataloader):

        data = batch[0].to(device)
        optimizer.zero_grad()
        # push whole batch of data through VAE to get recon_loss
        recon_batch, mu, logvar = model(data)
        # calculate scalar loss
        loss, bce, kld = model.loss_function(recon_batch, data, mu, logvar)
        # calculate the gradient of the loss w.r.t the graph leaves
        # i.e input variables
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        if batch_idx % 1 == 0:
            print('Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:0.6f}\tBCE: {:0.6f}\tKLD: {:0.6f}'.format(
                epoch, batch_idx, len(train_dataloader),
                100. * batch_idx / len(train_dataloader), loss.item(), bce.item(), kld.item()
            ))

        ###### !
        ###### !
        ###### !
        if batch_idx == l - 1:
            n = min(data.size(0), 8)
            # for the first batch of the epoch, show the first 8 input images
            # with right below them the reconstructed output images
            comparison = torch.cat([data[:n], recon_batch[:n]])
            save_image(comparison.cpu(), os.path.join(recon_dir, 'train_' + str(epoch) + '.png'), nrow=n)

            ###### !
            ###### !
            ###### !

            
    train_loss /= len(train_dataloader)
    print('====> Epoch: {} Average train loss: {:.4f}'.format(epoch, train_loss))
    return train_loss


@torch.no_grad()
def test(model, test_dataloader, epoch, device, recon_dir):
    # toggle model to test/inference mode
    model.eval()
    test_loss = 0
    for i, batch in enumerate(test_dataloader):
        
        data = batch.to(device)
        recon_batch, mu, logvar = model(data)
        loss, bce, kld = model.loss_function(recon_batch, data, mu, logvar)
        test_loss += loss.item()
        if i == 0:
            n = min(data.size(0), 8)
            # for the first batch of the epoch, show the first 8 input images
            # with right below them the reconstructed output images
            comparison = torch.cat([data[:n], recon_batch[:n]])
            save_image(comparison.cpu(), os.path.join(recon_dir, 'recon_' + str(epoch) + '.png'), nrow=n)

    test_loss /= len(test_dataloader) 
    print('====> Average test loss: {:.4f}'.format(test_loss))
    return test_loss



def main():

    ##############
    ### PARAMS ###
    ##############

    batch_train = 16
    batch_test = 16
    lr = 1e-4
    beta1 = 0.9
    epochs = 200
    latent_dim = 200
    nf = 64
    model_name = 'vae-64fc-200nz-deeper'
    val_every = 5
    load_path = './saved_models/best_acc-vae-64fc-200nz-deeper-cosine-choose_dict.pt'
    start_epoch = 1

    transform = T.Compose([T.Resize(size=256),
                           T.CenterCrop(size=224),
                           T.ToTensor()
    ])

    sample_dir = os.path.join('./results/samples/', model_name)
    recon_dir = os.path.join('./results/reconstructions', model_name)

    validate_params = {
        'per_row': 5,
        'metric': 'cosine',
        'model': None,
        'drop_last': None,
        'root_dir': '/media/STORAGE/DATASETS/open-images/train/',
        'csv_test': './csv_files/handpicked.csv',
        'csv_val': './csv_files/validation_labeled.csv',
        'batch_test': 1,
        'batch_val': 1,
        'save_test': True,
        'save_val': True,
        'vis_val': None,
        'vis_test': None,
        'algorithm': 'choose_dict'
    }

    ##############################
    ### DIRS FOR SAVING IMAGES ###
    ##############################

    try:
        os.makedirs(sample_dir)
    except FileExistsError:
        print('{} already exists!'.format(sample_dir))

    try:
        os.makedirs(recon_dir)
    except FileExistsError:
        print('{} already exists!'.format(recon_dir))

    ##############################
    ### DATASETS & DATALOADERS ###
    ##############################

    train_dataset_params = {
        'root': '/media/STORAGE/DATASETS/open-images/train/',
        'transform': transform,
    }

    test_dataset_params = {
        'csv_file': './csv_files/test.csv',
        'root_dir': '/media/STORAGE/DATASETS/open-images/challenge2018/',
        'transform': transform,
    }

    train_dataloader_params = {
        'batch_size': batch_train,
        'shuffle': True,
        'num_workers': 1
    }

    test_dataloader_params = {
        'batch_size': batch_test,
        'shuffle': True,
        'num_workers': 1
    }

    train_dataset = ImageFolder(**train_dataset_params)
    test_dataset = TestDataset(**test_dataset_params)

    train_dataloader = DataLoader(train_dataset, **train_dataloader_params)
    test_dataloader = DataLoader(test_dataset, **test_dataloader_params)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = VAE(latent_dim=latent_dim, nf=nf).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))

    best_train_loss = 99999999
    best_acc = 0
    if load_path:
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        start_epoch = epoch + 1
        best_train_loss = checkpoint['train_loss']
        best_acc = checkpoint['acc_val']
        print('Loaded model with train loss: {}, val_acc: {}'.format(checkpoint['train_loss'], checkpoint['acc_val']))

    #####################
    ### TRAINING LOOP ###
    #####################
    for epoch in range(start_epoch, epochs + start_epoch):
        train_loss = train(model, train_dataloader, optimizer, epoch, device, recon_dir)
        test_loss = test(model, test_dataloader, epoch, device, recon_dir)
        # train_loss = 10000
        # 64 sets of random latent_dim-float vectors, i.e 64 locations
        # in latent space
        sample = torch.randn(36, latent_dim).to(device)
        sample = model.decode(sample).detach().cpu()
        # save out as an 8x8 matrix
        # this will give you a visual idea of how well latent space can generate things
        save_image(sample, os.path.join(sample_dir, 'sample_' + str(epoch) + '.png'))

        ###########################
        ### CHARADES VALIDATION ###
        ###########################

        if epoch % val_every == 0: 
            alg = validate_params['algorithm']
            metric = validate_params['metric']
            validate_params['model'] = model.get_feature_extractor()
            validate_params['vis_val'] = './vis_val/' + model_name + '/epoch' + str(epoch) + '/' + metric + '-' + alg
            validate_params['vis_test'] = './vis_test/' + model_name + '/epoch' + str(epoch) + '/' + metric + '-' + alg
            acc_val, acc_test = validate(**validate_params)
            print(50 * '*')
            print('CHARADES ===> Val acc: {}, Test acc: {}'.format(acc_val, acc_test))
            print(50 * '*')

            if best_acc < acc_val:
                PATH = './saved_models/' + 'best_acc-' + model_name + '-' + metric + '-' + alg + '.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': best_train_loss,
                    'test_loss': test_loss,
                    'acc_val': best_acc,
                    'acc_test': acc_test
                    }, PATH) 
                print('Saving best acc model!')

        if train_loss < best_train_loss:
            alg = validate_params['algorithm']
            metric = validate_params['metric']
            PATH = './saved_models/' + 'best_train-' + model_name + metric + '-' + alg + '.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': best_train_loss,
                'test_loss': test_loss,
                'acc_val': best_acc,
                'acc_test': -1
                }, PATH)
            print('Saving best train loss model!')



if __name__ == '__main__':
    main()
    