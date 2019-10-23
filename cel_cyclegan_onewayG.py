import argparse
import os
import numpy as np
import itertools
import datetime
import time
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *
import torch.nn as nn
import torch.nn.functional as F
import torch
import calendar
from PIL import Image
from cityscapes import CityScapes

# import torchvision.models as torchvis_models
# from torchvision.utils import save_image


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="monet2photo", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
opt = parser.parse_args()


opt.dataset_name = 'cityscapes'
dt = datetime.datetime.today()
opt.experiment_name = opt.dataset_name+'-'+ calendar.month_abbr[dt.month]+'-'+str(dt.day)
opt.batch_size = 1
opt.batch_test_size = 1
opt.show_progress_every_n_iterations= 2  #  20  
opt.AMS_grad = True
opt.sample_interval = 100
opt.test_interval = 2 # 10
opt.checkpoint_interval = 2 #  10
opt.buffer_size_A = 50
opt.buffer_size_B = 50

opt.n_epochs = 50
opt.decay_epoch= 10 
opt.n_residual_blocks_A = 4
opt.n_residual_blocks_B = 9
opt.n_cpu = 4

opt.img_height = 100
opt.img_width = 200

# generate_all_test_images = True
# opt.seed_value =  12345 # np.random.randint(1, 2**32-1) 

opt.no_models = 34 #  19

print(opt)


# Create sample and checkpoint directories
os.makedirs('images/%s' % opt.experiment_name, exist_ok=True)
os.makedirs('saved_models/%s' % opt.experiment_name, exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda = torch.cuda.is_available()

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
# G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
G_AB = [ GeneratorResNet(input_shape, opt.n_residual_blocks_A).cuda() for i in range(opt.no_models)]
G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks_B)
# G_BA = [ GeneratorResNet(input_shape, opt.n_residual_blocks).cuda() for i in range(opt.no_models)] # in case one needs to do cellular for the backward model

D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)


if cuda:
    # G_AB = G_AB.cuda() # has already been used above
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

if opt.epoch != 0:
    # Load pretrained models
    ''' Rawi we will deal with this later  '''
    for jj in range(opt.no_models):
        G_AB[jj].load_state_dict(torch.load("saved_models/%s/G_AB_%d%s%d.pth" 
            % (opt.experiment_name, opt.epoch,'_', jj)))
    G_BA.load_state_dict(torch.load("saved_models/%s/G_BA_%d.pth" % (opt.experiment_name, opt.epoch)))
    D_A.load_state_dict(torch.load("saved_models/%s/D_A_%d.pth" % (opt.experiment_name, opt.epoch)))
    D_B.load_state_dict(torch.load("saved_models/%s/D_B_%d.pth" % (opt.experiment_name, opt.epoch)))
else:
    # Initialize weights
    for i in range(opt.no_models):
        G_AB[i].apply(weights_init_normal)
        #G_BA[i].apply(weights_init_normal) # in case one uses a backward cellular models

    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)
   
# Optimizers
optimizer_G = torch.optim.Adam(
      itertools.chain(nn.ModuleList(G_AB).parameters(), G_BA.parameters()),      
                                lr=opt.lr, betas=(opt.b1, opt.b2),
                                amsgrad=opt.AMS_grad) # amsgrad originally was false
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2),
                                 amsgrad=opt.AMS_grad)  # amsgrad originally was false
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2),
                                 amsgrad=opt.AMS_grad)  # amsgrad originally was false


# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer(max_size = opt.buffer_size_A)
fake_B_buffer = ReplayBuffer(max_size = opt.buffer_size_B)

''' if we want to take translation effect into account, we can 
      as well incldue reseize->crop etc '''
transform_data = transforms.Compose([   
       
# not needed as transform cty will do the scaling   transforms.Resize(( opt.img_height, opt.img_width), Image.NEAREST),    
#    transforms.Resize(int(opt.img_height * 1.12), Image.NEAREST),
#    transforms.RandomCrop((256, 256)),
#    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
]
)

transform_cty = transforms.Compose([    
   transforms.Resize(( opt.img_height, opt.img_width), Image.NEAREST)    
]
)


city_data = CityScapes(split='train', transform_cty = transform_cty,
                                    transform_data = transform_data)

city_data_val = CityScapes(split='val', transform_cty = transform_cty,  
                            transform_data = transform_data)


# Training data loader
dataloader = DataLoader(
    city_data,  
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers = opt.n_cpu,
)


val_dataloader = DataLoader(
    city_data_val,  
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers = opt.n_cpu,
)


def sample_images(batches_done):
    norm_imgs = False
    """Saves a generated sample from the test set"""
    # x_data = next(iter(val_dataloader))  # original
    x_data = next(iter(dataloader))  
    # classes_id = x_data["classes_id"]
    G_BA.eval()
    for jj in range(opt.no_models):  
       G_AB[jj].eval()           
    
    
    real_A = x_data["A"].type(Tensor)        
    
    with torch.no_grad(): 
       fake_A = G_BA(real_A) # again, real_B is real_A, hence we changed real_B to real_A
       
       fake_B_summed = torch.zeros(fake_A[0].size(), dtype=torch.float32,
                                         requires_grad=False).cuda()    
       for jj in range(opt.no_models):        
          fake_B = G_AB[jj](real_A)            
          fake_B_summed += fake_B.squeeze(dim=0)
          if jj== 0:          
             fake_B_0 = make_grid(fake_B, nrow=5, normalize=norm_imgs)
          elif jj==1:
             fake_B_1 = make_grid(fake_B, nrow=5, normalize=norm_imgs)
          elif jj==2:
             fake_B_2 = make_grid(fake_B, nrow=5, normalize=norm_imgs)    
                       
    # fake_B_summed = 2*(fake_B_summed-fake_B_summed.min())/(fake_B_summed.max()-fake_B_summed.min())-1
    # Making the grid 
    fake_B_summed = make_grid(fake_B_summed, nrow=5, normalize=norm_imgs)    
    real_A = make_grid(real_A, nrow=5, normalize= norm_imgs)
    fake_A = make_grid(fake_A, nrow=5, normalize=norm_imgs)    
    
    # Arange images along y-axis    
    image_grid = torch.cat((real_A, fake_B_summed, fake_A, fake_B_0, fake_B_1, fake_B_2), 1)
    save_image(image_grid, "images/%s/%s.png" % (opt.experiment_name, batches_done), normalize=False)
    
#    fake_B = make_grid(fake_B, nrow=5, normalize=True)
#    print("*************", fake_B.size)
#    image_grid = torch.cat( (fake_B), 1)
#    save_image(image_grid, "segments/%s/%s.png" % (opt.experiment_name, batches_done), normalize=False)

          
        

             
        
# ----------
#  Training the CycleGAN
# ----------

prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch  in enumerate(dataloader):
        
        # Set model input
        real_A = batch["A"] 
        real_A = Variable(real_A.type(Tensor)) # this should be one image                                      
        real_B = batch["B"].squeeze()  
        real_B = Variable(real_B.type(Tensor)) # this should be a total of no_classes  = 'no_models' images
        classes_id = batch["classes_id"]
        len_classes = torch.tensor(len(classes_id), requires_grad=False)
        range_class_id = range(len_classes)
        
        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        for jj in range_class_id: 
            G_AB[jj].train()
        G_BA.train()

        optimizer_G.zero_grad()                             
    
        # Identity loss   
        loss_id_B = 0              
        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        for jj, idx in enumerate(classes_id):             
           loss_id_B += criterion_identity(G_AB[idx](real_B[jj].unsqueeze(dim=0)), real_B[jj].unsqueeze(dim=0))
        loss_id_B = loss_id_B/ len_classes      
        loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN loss        
        
        loss_GAN_AB = 0 
        recov_B = 0
        fake_B_summed = torch.zeros(real_A.size(), dtype=torch.float32 ,
                                            requires_grad=True).cuda()
        recov_B = torch.zeros(real_A.size(), dtype=torch.float32 ,
                                            requires_grad=True).cuda()         
        fake_A = G_BA(real_A) # originally was fake_A = G_BA(real_B), but real_B == real_A if real_B images are superpositioned, seems like magic
        for jj, idx in enumerate(classes_id):   
           fake_B = ( G_AB[idx](real_A) )   
           fake_B_summed += fake_B
           loss_GAN_AB += criterion_GAN(D_B(fake_B ), valid) 
           recov_B += G_AB[idx](fake_A).squeeze(dim=0)
        loss_GAN_AB = loss_GAN_AB/len_classes        
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss        
        # First, A cycle loss # recov_A = G_BA(fake_B)               
        recov_A = G_BA(fake_B_summed)
        loss_cycle_A = criterion_cycle(recov_A, real_A)                 
        loss_cycle_B = criterion_cycle(recov_B, real_A) # again, we are changint real_B to real_A; as real_B == real_A, seems like magic                
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total loss
        loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity

        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        optimizer_D_A.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_A(real_A), valid)        
        fake_A_ = fake_A_buffer.push_and_pop(fake_A) # Fake loss (on batch of previously generated samples)
        loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        loss_D_A.backward()
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        optimizer_D_B.zero_grad()        
        
        loss_real = 0
        for jj in range_class_id:
           loss_real += criterion_GAN(D_B(real_B[jj].unsqueeze(dim=0)), 
                                      valid)
        loss_real= loss_real/len_classes
        # Fake loss (on batch of previously generated samples)
        
        fake_B_ = fake_B_buffer.push_and_pop(fake_B_summed)  # replaced fake_B with fake_B_summed        
        loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        loss_D_B.backward()
        optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f]\
             [G loss: %f, adv: %f, cycle: %f, identity: %f]\
             [loss_idA: %f, loss_idB: %f, lossG_A: %f, loss_G_B: %f,\
             lossC_A: %f, loss_C_B: %f]\
             ETA: %s "
             
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_GAN.item(),
                loss_cycle.item(),
                loss_identity.item(),
                loss_id_A.item(),
                loss_id_B.item(),
                loss_GAN_AB.item(),
                loss_GAN_BA.item(),
                loss_cycle_A.item(),
                loss_cycle_B.item(),
                time_left,
            )
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
           sample_images(batches_done)
           # print('..', end="")
           

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        for jj in range(opt.no_models):
                torch.save(G_AB[jj].state_dict(), 
                       "saved_models/%s/G_AB_%d%s%d.pth" % (opt.experiment_name, opt.epoch,'_',jj)) 
        torch.save(G_BA.state_dict(), 'saved_models/%s/G_BA_%d.pth' % (opt.experiment_name, epoch))
        torch.save(D_A.state_dict(), 'saved_models/%s/D_A_%d.pth' % (opt.experiment_name, epoch))
        torch.save(D_B.state_dict(), 'saved_models/%s/D_B_%d.pth' % (opt.experiment_name, epoch))
        
    


#def img_pooling(fake_A, fake_B, classes_id, pool_mode='sum' ): #  pool_mode=sum or max
#
#   if pool_mode == 'sum':
#      fake_B_summed = torch.zeros(fake_B[0].size(), dtype=torch.float32 ,
#                                            requires_grad=True).cuda()
#      recov_B = torch.zeros(fake_A.size(), dtype=torch.float32 ,
#                                            requires_grad=True).cuda()      
#      for jj, idx in enumerate(classes_id):            
#         fake_B_summed += fake_B[jj] # this is someting like average pooling # after sum, we will have values >1 and <-1 (could reach -7 to 7, or higher), hence, we normalize to bound the tensor between [-1, 1]                     
#         recov_B += G_AB[idx](fake_A).squeeze(dim=0)
#      
#      #fake_B_summed = 2*(fake_B_summed-fake_B_summed.min())/(fake_B_summed.max()-fake_B_summed.min())-1
#      
#      # recov_B = 2*(recov_B-recov_B.min())/(recov_B.max()-recov_B.min())-1             
#      
#   else: 
#      fake_B_summed = -100*torch.ones(fake_B[0].size(), dtype=torch.float32 ,
#                                         requires_grad=True).cuda()
#      recov_B = -100*torch.ones(fake_A.size(), dtype=torch.float32,
#                                   requires_grad=True).cuda()
#        
#      for jj, idx in enumerate(classes_id):            
#         fake_B_summed = torch.max(fake_B_summed, fake_B[jj]) # this is someting like average pooling # after sum, we will have values >1 and <-1 (could reach -7 to 7, or higher), hence, we normalize to bound the tensor between [-1, 1]              
#         recov_B = torch.max(G_AB[idx](fake_A).squeeze(dim=0), recov_B)
#   
#   return fake_B_summed, recov_B
#              