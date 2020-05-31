import torch
import torchvision



class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    """
    Custom dataset that includes image path when returning sample.
    Returns:
        (image, label, path)
    """
    
    def __getitem__(self, index):
        (image, label) = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        return (image, label, path)


if __name__ == '__main__':

    transform = torchvision.transforms.ToTensor()
    data_dir  = '~/datasets/open-images/train/'
    dataset = ImageFolderWithPaths(data_dir, transform)
    dataloader = torch.utils.data.DataLoader(dataset)



    # for images, labels, paths in dataloader:
    #     print(images, labels, paths)

