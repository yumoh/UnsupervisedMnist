import torchvision.datasets
import torchvision.transforms as T
import torch.utils.data


class Mnist:
    def __init__(self, batch_size=32):
        ts = T.Compose([
            T.Resize(32),
            T.ToTensor()
        ])
        train_data = torchvision.datasets.MNIST('data', train=True, transform=ts)
        validation_data = torchvision.datasets.MNIST('data', train=False, transform=ts)

        self.train_loader = torch.utils.data.DataLoader(train_data,
                                                        batch_size=batch_size,
                                                        shuffle=True
                                                        )
        self.validation_loader = torch.utils.data.DataLoader(validation_data,
                                                             batch_size=batch_size,
                                                             shuffle=True
                                                             )
