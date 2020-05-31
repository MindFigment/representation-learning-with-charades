import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


# https://vxlabs.com/2017/12/08/variational-autoencoder-in-pytorch-commented-and-annotated/

CUDA = True
SEED = 1
BATCH_SIZE = 10
LOG_INTERVAL = 10
EPOCHS = 10

# connections through the autoencoder bottleneck
ZDIMS = 20

torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)

# DataLoader instances will load tensors directly input GPU memory
kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

# Load train data
# shuffle at every epoch
train_dataloader = None


# Load test data
# shuffle at every epoch
test_dataloader = None


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()

        ###############
        ### ENCODER ###
        ###############

        # 224 x 224 x 3 input pixels, 400 outputs
        self.fc1 = nn.Linear(224 * 224 * 3, 400)
        self.relu = nn.ReLU()
        self.fc21 = nn.Linear(400, ZDIMS)
        self.fc22 = nn.Linear(400, ZDIMS)

        ###############
        ### DECODER ###
        ###############

        self.fc3 = nn.Linear(ZDIMS, 400)
        self.fc4 = nn.Linear(400, 224 * 224 * 3)
        self.sigmoid = nn.Sigmoid()

    
    def encode(self, x: Variable) -> (Variable, Variable):
        """
        Input vector x-> fully connected 1 -> ReLu ->
        (fully connected 21, fully connected 22)

        Parameters
        ----------
        x: [None, 224 * 224 * 3]

        Returns
        -------

        (mu, logvar): zdims mean units one for each latent dimension, zdims
            variance units one for each latent dimension
        
        """

        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    
    def reparameterize(self, mu: Variable, logvar: Variable) -> Variable:
        """
        THE REPARAMETERIZATION IDEA

        For each training sample:

            - take current learned mu, stddev for each of the ZDIMS
              dimensions and draw a random sample from that distribution
            - the whole network is trained so that these randomly drawn 
              samples decode to output that looks like the input
            - which will mean that std, mu will be learned
              *distributions* that correctly encode the inputs
            - due to the additional KLD term the distribution will
              tend to unit Gaussians

        Parameters
        ----------
        mu: [None, ZDIMS] -> mean matrix
        logvar: [None, ZDIMS] -> variance matrix

        Returns
        -------
        During training returns random sample from the learned ZDIMS-dimensional
        normal distribution.
        During inference returns its mean.

        """

        if self.training:
            # multiply log variance with 0.5, then in-place exponent
            # yielding the standard devation
            std = logvar.mul(0.5).exp_()
            # std.data is the [None, ZDIMS] tensor that is wrapped by std
            # so eps is [None, ZDIMS] with all elemennts drawn from a mean 0
            # and stddev 1 normal distribution that is batch of samples
            # of random ZDIMS-float vectors
            eps = Variable(std.data.new(std.size()).normal_())
            # sample from a normal distribution with standard
            # devation = std and mean = mu b multiplying mean 0
            # stddev 1 samle with desired std and mu, see
            # https://stats.stackexchange.com/a/16338
            # so we have a batch of random ZDIMS-float vectors
            # sampled from normal distribution with learned
            # std and mu for the current input
            return eps.mul(std).add_(mu)

        else:
            # During inference, we simply split out the mean of the
            # learned distribution for the current input. We could
            # use a random sample from the distribution, but mu of
            # course has the highest probability
            return mu

        
    def decode(self, z: Variable) -> Variable:
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    
    def forward(self, x: Variable) -> (Variable, Variable, Variable):
        mu, logvar = self.encode(x.view(-1, 224 * 224 * 3))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE()
if CUDA:
    model.cuda()


def loss_function(recon_x, x, mu, logvar) -> Variable:
    # how well do input x and output recon_x agree?
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 224 * 224 * 3))

    # KLD is Kullback-Leibler divergence: how much does one learned
    # distribution deviate from another, in this specific case the
    # learned distribution from the unit Gaussian

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # note the negative D_{KL} in appendix B of the paper
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalize by same number of elements as in reconstruction
    KLD /= BATCH_SIZE * (224 * 224 * 3)

    # BCE tries to make our reconstruction as accurate as possible
    # KLD tries to push the distributions as close as possible to unit Gaussian
    return BCE + KLD


optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
    # toogle model to train mode
    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(train_dataloader):
        data = Variable(batch)
        if CUDA:
            data = data.cuda()
        optimizer.zero_grad()

        # push whole batch of data through VAE.forward() to get recon_loss
        recon_batch, mu, logvar = model(data)
        # calculate scalar loss
        loss = loss_function(recon_batch, data, mu, logvar)
        # calculate the gradient of the loss w.r.t the graph leaves
        # i.e input variables
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:0.6f}'.format(
                epoch, batch_idx * len(data), len(train_dataloader.dataset),
                100. * batch_idx / len(train_dataloader),
                loss.data[0] / len(data)
            ))
            
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_dataloader.dataset)
    ))


def test(epoch):
    # toggle model to test/inference mode
    model.eval()
    test_loss = 0

    # each data is of BATCH_SIZE samples
    for i, batch in enumerate(test_dataloader):
        if CUDA:
            # make sure this lives on the GPU
            data = batch.cuda()
        
        # we're only going to infer, so no autograd at all required: volatile=True
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
        if i == 0:
            n = min(data.size(0), 8)
            # for the first batch of the epoch, show the first 8 input images
            # with rightt below them the reconstructed output digits
            comparison = torch.cat([data[:n],
                                    recon_batch.view(BATCH_SIZE, 3, 224, 224)[:n]])
            save_image(comparison.data.cpu(),
                       'results/reconstructions/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_dataloader.dataset)
    print('=====> Test set loss: {:.4f}'.format(test_loss))

                       

for epoch in range(1, EPOCHS + 1):
    train(epoch)
    test(epoch)

    # 64 sets of random ZDIMS-float vectors, i.e 64 locations
    # in latent space
    sample = Variable(torch.randn(64, ZDIMS))
    if CUDA:
        sample = sample.cuda()
    sample = model.decode(sample).cpu()

    # save out as an 8x8 matrix
    # this will give you a visual idea of how well latent space can generate things
    save_image(sample.data.view(64, 3, 224, 224),
               'results/samples/sample_' + str(epoch) + '.png')

