import numpy as np 
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
class Trainer():
    def __init__(self, net, loss_func, optim):
        # super(Trainer, self).__init__()
        self.net = net
        self.loss_func = loss_func 
        self.optim = optim 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, dataloader, epoch):
        self.net.train()  #Put the network in train mode
        total_loss = 0
        batches = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = Variable(data), Variable(target)
            batches += 1
            data = data.to(self.device)
            # Training loop
            pred = self.net(data)
            data = data.cpu()
            target = target.to(self.device)
            loss = self.loss_func(pred,target)
            self.net.zero_grad() 
            self.optim.zero_grad() 
            loss.backward() 
            self.optim.step()


            total_loss += loss
            if batch_idx % 100 == 0: #Report stats every x batches
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, (batch_idx+1) * len(data), len(dataloader.dataset),
                            100. * (batch_idx+1) / len(dataloader), loss.item()), flush=True)
        av_loss = total_loss / batches
        av_loss = av_loss.detach().cpu().numpy()

        print('\nTraining set: Average loss: {:.4f}'.format(av_loss,  flush=True))

        #print('Time taken for epoch = ', total_time)
        return av_loss

    def val(self, val_dataloader, epoch):
        self.net.eval()  #Put the model in eval mode
        total_loss = 0    
        batches = 0
        with torch.no_grad():  # So no gradients accumulate
            for batch_idx, (data, target) in enumerate(val_dataloader):
                batches += 1
                data, target = Variable(data), Variable(target)
                data = data.to(self.device)
                # Eval steps
                pred = self.net(data)
                data = data.cpu()
                target = target.to(self.device)
                loss = self.loss_func(pred,target)

                total_loss += loss
                
            av_loss = total_loss / batches
            
        av_loss = av_loss.detach().cpu().numpy()
        print('Validation set: Average loss: {:.4f}'.format(av_loss,  flush=True))
        print('\n')
        return av_loss

    def do_training(self, train_dataloader, val_dataloader, max_epochs):
        sum = 0 
        for param in self.net.parameters():
            sum += param.numel() 
        print("Trainable params: ", sum)

        losses = [] 
        for epoch in range(1, max_epochs+1):
            train_loss = self.train(train_dataloader, epoch)
            val_loss = self.val(val_dataloader, epoch)
            losses.append([train_loss, val_loss])

        losses = np.array(losses).T
        print(losses.shape)
        its = np.linspace(1, max_epochs, max_epochs)

        plt.figure()
        plt.plot(its, losses[0,:])
        plt.plot(its, losses[1,:])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'])

        from datetime import datetime

        now = datetime.now()

        current_time = now.strftime("%H_%M_%S")
        plt.savefig(current_time + ".png")

    
    
        return self.net 
