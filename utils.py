import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from PIL import Image
import pickle
import time
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torch.nn import BCEWithLogitsLoss, L1Loss, MSELoss
from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet
from tqdm import tqdm

# set paths
images_path = os.path.join(os.getcwd(), 'images') 
weights_path = os.path.join(os.getcwd(), 'weights') 
history_path = os.path.join(os.getcwd(), 'history') 
plots_path = os.path.join(os.getcwd(), 'plots') 
# create folders if they do not already exist
if not os.path.exists(images_path): os.makedirs(images_path)
if not os.path.exists(weights_path): os.makedirs(weights_path)
if not os.path.exists(history_path): os.makedirs(history_path)
if not os.path.exists(plots_path): os.makedirs(plots_path)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set data transformation dictionary for training, validation
data_transforms = {
    'train': T.Compose([
        # to complete, for now it does not seem to work very well so I commented it
        T.Resize((286,286)),
        T.RandomCrop((256,256))
    ]),
    'val': T.Compose([
    ])
}


def get_train_test_image_names(set, images_path = images_path):
    """
    Description
    -------------
    List names of train and test videos
    Parameters
    -------------
    dataset_name   : name of dataset
    labels_path    : path to datasets 
    Returns
    -------------
    (train_names, val_names) , each of the tuple elements is a list of strings corresponding to images names
    """
    images_names = {}
    images_path = os.path.join(images_path, set) 
    
    for mode in ['train','val']:
        # list image names and 
        images_names[mode] = os.listdir(os.path.join(images_path, mode))
        images_names[mode].sort()

    return images_names


class ImageDataset(Dataset):
    """Instance of the image dataset."""

    def __init__(self, dataset_name, transform = None, mode = 'train'):
        """
        Description
        -------------
        Creates dataset class for the training set.
        Parameters
        -------------
        dataset_name            : name of dataset
        transform               : transforms to be applied to the frame (eg. data augmentation)
        mode                    : string, either 'train' or 'val'
        Returns
        -------------
        Torch Dataset for the training set
        """
        # set class parameters
        self.images_folder_path = images_path + '/' + dataset_name + '/' + mode
        self.transform = transform
        # set image names
        self.images_names = get_train_test_image_names(dataset_name)[mode]

    def __len__(self):
        # return length of the list of image names
        return len(self.images_names)

    def __getitem__(self, index):
        # recover image path
        image_path =  self.images_folder_path + '/' + self.images_names[index] 
        # read image
        image = Image.open(image_path)
        # convert to tensor
        image = T.functional.to_tensor(image)
        # recover real and input image
        image_width = image.shape[2]
        real_image = image[:, :, : image_width // 2]
        input_image = image[:, :, image_width // 2 :]

        if self.transform:
            # apply data transformation
            real_image = self.transform(real_image)
            input_image = self.transform(input_image)
            # random mirroring
            input_image = torch.flip(input_image, [2])
            real_image = torch.flip(real_image, [2])

        return input_image, real_image

def generate_images(model, input, real):
    """
    Description
    -------------
    Plot input, real image and model predictions side by side
    Parameters
    -------------
    input       : input image
    model       : Pix2Pix model
    """
    prediction = model.generator(input.to(device))

    # create figure
    plt.figure(figsize=(10,5))

    # recover image of each batch of size 1
    display_list = [input[0], real[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot
        to_display = np.clip(display_list[i].permute(1,2,0).detach().cpu().numpy(), 0, 1)
        plt.imshow(to_display)
        plt.axis('off')
    plt.show()
    
    
def generate_images_multimodel(model_list, input_list, real_list, forced_title = False):
    """
    Description
    -------------
    Plot input, real image and model predictions side by side for several models at the same time
    Parameters
    -------------
    model_list  : list of Pix2Pix models 
    input_list  : list of input images
    real_list   : list of real images
    force_title : for report purpose, to have handcrafted titles for plots
    """

    # create figure
    plt.figure(figsize=(5 + 4.3 * len(model_list),4 * len(input_list)))

    # recover image of each batch of size 1
    display_list = []
    title = []
    for input, real in zip(input_list, real_list):
        predictions = [] #predictions for each model
        for model in model_list:
            predictions.append(model.generator(input.to(device)))

        display_list += [input[0], real[0]]
        display_list += [prediction[0] for prediction in predictions]
        title.extend(['Input Image', 'Ground Truth'])
        if forced_title:
            title.extend(['Original Pix2Pix (our implementation)', 'Our new Pix2Pix on 10 epochs', 'Our new Pix2Pix on 20 epochs'])
        else:
            title.extend([f'Predicted Image by model {i}' for i in range(len(model_list))])

    n_images = (2+ len(model_list)) * len(input_list)
    for i in range(n_images):
        plt.subplot(len(input_list), 2 + len(model_list), i+1)
        if i>= n_images - 5:
            plt.title(title[i], y=-0.14)
        # Getting the pixel values in the [0, 1] range to plot
        to_display = np.clip(display_list[i].permute(1,2,0).detach().cpu().numpy(), 0, 1)
        plt.imshow(to_display)
        plt.axis('off')
    plt.show()
    
   
def generate_color_histogram(model_list, input, real):
    """
    Description
    -------------
    Generates color distributions for several models on an input image
    Parameters
    -------------
    input   : input image
    real    : real image
    model_list  : list of Pix2Pix models
    """
    # create figure
    plt.figure(figsize=(5 + 5 * len(model_list),5))
    
    #Compute predictions 
    predictions = [] #predictions for each model
    for model in model_list:
        predictions.append(model.generator(input.to(device)))

    # recover image of each batch of size 1
    display_list = [input[0], real[0]] + [prediction[0] for prediction in predictions]

    title = ['Input Image', 'Ground Truth', 'Original Pix2Pix (our implementation)', 'Our new Pix2Pix on 10 epochs', 'Our new Pix2Pix on 20 epochs']

    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot
        to_display = np.clip(display_list[i].permute(1,2,0).detach().cpu().numpy(), 0, 1)
        plt.imshow(to_display)
        plt.axis('off')
    plt.show()

    styles = [':', '--', '-']
    plt.figure(figsize=(20,10))
    plt.subplot(3,1,1)
    to_display = 256*real[0].permute(1,2,0).detach().cpu().numpy()
    blue_histogram = cv2.calcHist([to_display], [0], None, [256], [0, 256])
    plt.plot(blue_histogram, label = 'Ground Truth', c = 'darkblue')

    

    for j, prediction in enumerate(predictions):
        to_display = 256*prediction[0].permute(1,2,0).detach().cpu().numpy()
        blue_histogram = cv2.calcHist([to_display], [0], None, [256], [0, 256])
        plt.plot(blue_histogram, label = title[2+j], ls = styles[j], c = 'black')
    plt.legend(loc="upper right")
    plt.title("blue")

    plt.subplot(3,1,2)
    to_display = 256*real[0].permute(1,2,0).detach().cpu().numpy()
    red_histogram = cv2.calcHist([to_display], [1], None, [256], [0, 256])
    plt.plot(red_histogram, label = 'Ground Truth', c = 'red')
    
    for j, prediction in enumerate(predictions):
        to_display = 256*prediction[0].permute(1,2,0).detach().cpu().numpy()
        red_histogram = cv2.calcHist([to_display], [1], None, [256], [0, 256])
        plt.plot(red_histogram, label = title[2+j], ls = styles[j], c = 'black')
    plt.legend(loc="upper right")
    plt.title("red")

    plt.subplot(3,1,3)
    to_display = 256*real[0].permute(1,2,0).detach().cpu().numpy()
    green_histogram = cv2.calcHist([to_display], [2], None, [256], [0, 256])
    plt.plot(green_histogram, label = 'Ground Truth', c = 'green')
    
    for j, prediction in enumerate(predictions):
        to_display = 256*prediction[0].permute(1,2,0).detach().cpu().numpy()
        green_histogram = cv2.calcHist([to_display], [2], None, [256], [0, 256])
        plt.plot(green_histogram, label = title[2+j], ls = styles[j], c = 'black')
        plt.legend(loc="upper right")
    plt.title("green")
    plt.show()
 

def train(model, n_epochs, display_step, save_step, dataloaders, filename, lr = 2e-4, lbd = 200, loss_l1_true = True, loss_l2_true = False, loss_cGAN_true = True):
    """
    Training of the Pix2Pix model by firstly trainin the generator and then training the discriminator
    """

    """
    Description
    -------------
    Train the Pix2Pix model.
    Parameters
    -------------
    model                   : model to train
    n_epochs                : number of epochs to train the model on
    display_step            : number of epochs between two displays of images
    save_step               : number of epochs between two saves of the model
    dataloaders             : dataloader to use for training
    filename                : string, a filename to give to weights
    lr                      : learning rate
    lbd                     : l1 and l2 loss weight
    loss_l1_true            : bool to take into account l1 loss in the generator loss
    loss_l2_true            : bool to take into account l2 loss in the generator loss
    loss_cGAN_true          : bool to take into account cGAN loss in the generator loss
    Returns
    -------------
    History of training (dict)
    """
    def compute_gen_loss(real_images, conditioned_images):
        """ Compute generator loss. """
        # compute adversarial loss
        fake_images = model.generator(conditioned_images)
        disc_logits = model.discriminator(fake_images, conditioned_images)
        adversarial_loss = BCEWithLogitsLoss()(disc_logits, torch.ones_like(disc_logits))
        # compute reconstruction loss l1
        recon_loss_l1 = L1Loss()(fake_images, real_images)
        # compute reconstruction loss l2
        recon_loss_l2 = MSELoss()(fake_images, real_images)
        # compute the generator loss
        loss_gen = loss_cGAN_true*adversarial_loss + lbd * (loss_l1_true*recon_loss_l1+loss_l2_true*recon_loss_l2)
        return loss_gen , adversarial_loss, recon_loss_l1, recon_loss_l2

    def compute_disc_loss(real_images, conditioned_images):
        """ Compute discriminator loss. """
        fake_images = model.generator(conditioned_images).detach()
        fake_logits = model.discriminator(fake_images, conditioned_images)

        real_logits = model.discriminator(real_images, conditioned_images)

        fake_loss = BCEWithLogitsLoss()(fake_logits, torch.zeros_like(fake_logits))
        real_loss = BCEWithLogitsLoss()(real_logits, torch.ones_like(real_logits))
        return (real_loss + fake_loss) / 2

    # compute dataset length
    n = len(dataloaders['train'])

    # initalize optimizers
    optimizer_G = torch.optim.Adam(model.generator.parameters(), lr=lr)
    optimizer_D = torch.optim.Adam(model.discriminator.parameters(), lr=lr)

    # initialize timer
    since = time.time()

    # switch to eval mode and disable grad tracking
    model.eval()
    with torch.no_grad():
        input_val, real_val = next(iter(dataloaders['val']))
        generate_images(model = model, input = input_val, real = real_val)
    # switch back to train mode
    model.train()

    # instantiate history array
    history = {'gen_loss' : [], 'gan_loss' : [], 'l1_loss' : [],'l2_loss': [], 'disc_loss' : []}

    for epoch in range(n_epochs):
        print(f'Epoch {epoch + 1}/{n_epochs}')
        print('-' * 10)

        # set epoch losses to 0
        epoch_gen_loss = 0
        epoch_gan_loss = 0
        epoch_l1_loss = 0
        epoch_l2_loss = 0
        epoch_disc_loss = 0

        for input, real in dataloaders['train']:
            # send tensors to device
            input = input.to(device)
            real = real.to(device)

            # generator step
            optimizer_G.zero_grad()
            gen_loss, gan_loss, l1_loss, l2_loss = compute_gen_loss(real, input)
            gen_loss.backward()
            optimizer_G.step()

            # discriminator step
            optimizer_D.zero_grad()
            disc_loss = compute_disc_loss(real, input)
            disc_loss.backward()
            optimizer_D.step()

            # update epoch losses
            epoch_gen_loss += gen_loss
            epoch_gan_loss += gan_loss
            epoch_l1_loss += l1_loss
            epoch_l2_loss += l2_loss
            epoch_disc_loss += disc_loss
            
        # print losses
        print(f'gen_loss: {epoch_gen_loss/n:.4f}, gan_loss: {epoch_gan_loss/n:.4f}, l1_loss: {epoch_l1_loss/n:.4f}, disc_loss: {epoch_disc_loss/n:.4f}.')
        history['gen_loss'].append(epoch_gen_loss.item()/n)
        history['gan_loss'].append(epoch_gan_loss.item()/n)
        history['l1_loss'].append(epoch_l1_loss.item()/n)
        history['l2_loss'].append(epoch_l2_loss.item()/n)
        history['disc_loss'].append(epoch_disc_loss.item()/n)


        if (epoch + 1) % display_step == 0:
            # switch to eval mode and disable grad tracking
            model.eval()
            with torch.no_grad():
                generate_images(model = model, input = input_val, real = real_val)
            # switch back to train mode
            model.train()

        if (epoch + 1) % save_step ==0:
            # save model weights
            print('saving model weights ...')
            torch.save(model.state_dict(), weights_path + '/' + filename + '_ep' + str(epoch) + '.pkl')

        # line break for better readability
        print('\n')

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
    # save history 
    history_file = open(history_path + '/' + filename + '.pkl', "wb")
    pickle.dump(history, history_file)
    history_file.close()
    
    return history

    
    
def plot_and_save_history(history, filename, title = None):
    """
    Description
    -------------
    Plots and saves history
    Parameters
    -------------
    history     : a dictionary of metrics to plot with keys 'gen_loss', 'gan_loss', 'l1_loss' and 'disc_loss'
    filename    : string, a filename 
    title       : title for the plot
    """

    # save history 
    history_file = open(history_path + '/' + filename + '.pkl', "wb")
    pickle.dump(history, history_file)
    history_file.close()

    fig, axs = plt.subplots(2, 2, figsize=(8,7))

    colors = ['blue', 'green', 'red', 'purple']

    # plot
    for i, ax in enumerate(axs.reshape(-1)):
        ax.grid()
        ax.set_xlabel('epoch')
        ylab = list(history.keys())[i]
        ax.set_ylabel(ylab)
        ax.plot(history[ylab], color = colors[i])

    # add title
    if title:
        plt.subplots_adjust(wspace=0.5)
        plt.suptitle(title)
    
    # save plot and show
    plt.savefig(history_path + '/' + filename + '.png')
    plt.show()
 
#Below: for our enhanced pix2pix on colorization task


def build_res_unet(n_input=3, n_output=3, size=256):
    """
    Description
    -------------
    Builds the downsampling backbone for our new generator; instead of training it, we load weights 
    from a Resnet-18 trained on ImageNet, a way bigger dataset that ours.
    Parameters
    -------------
    n_input     : number of channels (e.g. r,g,b: 3) of our input
    n_output    : number of channels (e.g. r,g,b: 3) of our output
    size        : size of images (here: 256x256)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = create_body(resnet18, pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)
    return net_G

def pretrain_generator(net_G, dataloader, epochs = 10):
    """
    Description
    -------------
    Pretrains the full generator on our dataset for minimizing the L1 loss on the colorization task;
    this is done before putting the Generator against the discriminator, to give our model some sense
    of what it needs to do before starting the adversarial process.
    Parameters
    -------------
    net_G       : pre-trained generator on Resnet-18
    dataloader  : our dataloader
    epochs      : number of epochs on the colorization task
    """
    opt = torch.optim.Adam(net_G.parameters(), lr=1e-4)
    criterion = torch.nn.L1Loss()
    for e in range(epochs):
        loss_meter = AverageMeter()
        for input, real in tqdm(dataloader['train']):
            # send tensors to device
            input = input.to(device)
            real = real.to(device)
            preds = net_G(input)
            loss = criterion(preds, real)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            loss_meter.update(loss.item(), input.size(0))
            
        print(f"Epoch {e + 1}/{epochs}")
        print(f"L1 Loss: {loss_meter.avg:.5f}")

class AverageMeter:
    """
    Description
    -------------
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3
    
    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count
