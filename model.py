import torch
import torch.nn as nn

class Generator(nn.Module):
    """ Generator of the Pix2Pix model. """

    def __init__(self):
        """
        Description
        -------------
        Initialize generator model
        """
        super(Generator, self).__init__()

        # instantiate downsampling layers
        self.downsample_layers = nn.ModuleList([self.__create_downsample_layer(3, 64, apply_batchnorm=False),
                                self.__create_downsample_layer(64, 128),
                                self.__create_downsample_layer(128, 256),
                                self.__create_downsample_layer(256, 512),
                                self.__create_downsample_layer(512, 512),
                                self.__create_downsample_layer(512, 512),
                                self.__create_downsample_layer(512, 512),
                                self.__create_downsample_layer(512, 512, apply_batchnorm=False)]) # to modify

        # instantiate upsampling layers
        self.upsample_layers = nn.ModuleList([self.__create_upsample_layer(512, 512, apply_dropout=True),
                                self.__create_upsample_layer(1024, 512, apply_dropout=True),
                                self.__create_upsample_layer(1024, 512, apply_dropout=True),
                                self.__create_upsample_layer(1024, 512),
                                self.__create_upsample_layer(1024, 256),
                                self.__create_upsample_layer(512, 128),
                                self.__create_upsample_layer(256, 64)])
        
        # create last layer
        self.last = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False)

    def __create_downsample_layer(self, in_channels, out_channels, kernel_size=4, padding = 1, apply_batchnorm = True):
        """
        Description
        -------------
        Creates downsample layer with convolution, batchnorm and activation
        """
        layers = []
        conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=padding, bias=False)
        layers.append(conv_layer)
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU())
        return nn.Sequential(*layers)
        
    def __create_upsample_layer(self, in_channels, out_channels, kernel_size=4, padding=1, apply_dropout = False):
        """
        Description
        -------------
        Creates upsample layer with deconvolution, batchnorm, dropout and activation
        """
        layers = []
        deconv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, padding=padding, bias=False)
        layers.append(deconv_layer)
        layers.append(nn.BatchNorm2d(out_channels))
        if apply_dropout:
            layers.append(nn.Dropout(p=0.5))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)
    

    def forward(self, x):
        """
        Description
        -------------
        Forward pass
        Parameters
        -------------
        x                : tensor of shape (batch_size, c, w, h)
        """
        
        # downsampling phase
        skips = []
        for down in self.downsample_layers:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # upsampling while concatenating skip filter maps
        for up, skip in zip(self.upsample_layers, skips):
            x = up(x)
            x = torch.cat((x, skip), 1)

        # apply last layer
        x = self.last(x)
        x = nn.Tanh()(x)
        
        return x

class Discriminator(nn.Module):
    """ Discriminator of the Pix2Pix model. """

    def __init__(self):
        """
        Description
        -------------
        Initialize discriminator model
        """
        super(Discriminator, self).__init__()

        # instantiate downsampling layers
        self.down1 = self.__create_downsample_layer(6, 64, 4, apply_batchnorm = False)
        self.down2 = self.__create_downsample_layer(64, 128, 4)
        self.down3 = self.__create_downsample_layer(128, 256, 4)
        # instantiate other layers
        self.conv = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, bias=False)
        self.batchnorm = nn.BatchNorm2d(512)
        self.last = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1)

    def __create_downsample_layer(self, in_channels, out_channels, kernel_size=4, padding = 1, apply_batchnorm = True):
        """
        Description
        -------------
        Creates downsample layer with convolution, batchnorm and activation
        """
        layers = []
        conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=padding, bias=False)
        layers.append(conv_layer)
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def set_requires_grad(self, bool):
        for param in self.parameters():
            param.requires_grad = bool


    def forward(self, input, real):
        """
        Description
        -------------
        Forward pass
        Parameters
        -------------
        input               : input image, tensor of shape (batch_size, c, w, h)
        real                : real target image, tensor of shape (batch_size, c, w, h)
        """
        x = torch.cat([input, real], 1) # concatenate along channels
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = nn.ZeroPad2d(1)(x)
        x = self.conv(x)
        x = self.batchnorm(x)
        x = nn.LeakyReLU()(x)
        x = nn.ZeroPad2d(1)(x)
        x = self.last(x)

        return x

class Pix2Pix(nn.Module):
    """
    The Pix2Pix archiecture with U-Net generator and patch GAN disciminator
    """
    def __init__(self, generator = None):
        super(Pix2Pix, self).__init__()
        if generator is None: #in pix2pix by default
            self.generator = Generator()
            self.generator = self.generator.apply(self._weights_init) # intialize weights
        else: #allows us to plug a custom generator in pix2pix
            self.generator = generator.to(self.device)
        self.discriminator = Discriminator()
        # intialize weights
        self.discrimator = self.discriminator.apply(self._weights_init)

    def _weights_init(self, m):
        """
        Description
        -------------
        Weights initialization method
        """
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)
