"""
Computer Vision Final Project 
Author: Juo-Tung Chen

"""

import torch.optim as optim
import torch.utils.data
import matplotlib
# matplotlib.use('Agg')  # or 'agg' depending on your preference
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.models as models
import torch.nn.functional as F
from torchvision import transforms as transforms
import numpy as np
import os
from PIL import Image

import argparse

from models import *
from misc import progress_bar

from cnn_finetune import make_model

class CarDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images, self.labels = self.read_images_and_labels(self.root_dir)
        self.CLASSES = list(set(self.labels))
        self.image_paths = self.get_image_paths()

    def get_image_paths(self):
        image_paths = []
        for make in os.listdir(self.root_dir):
            make_path = os.path.join(self.root_dir, make)
            if os.path.isdir(make_path):
                for model in os.listdir(make_path):
                    model_path = os.path.join(make_path, model)
                    if os.path.isdir(model_path):
                        for generation in os.listdir(model_path):
                            generation_path = os.path.join(model_path, generation)
                            if os.path.isdir(generation_path):
                                for filename in os.listdir(generation_path):
                                    image_paths.append(os.path.join(generation_path, filename))
        return image_paths

    def read_images_and_labels(self, dataset_path):
        images = []
        labels = []

        for make in os.listdir(dataset_path):
            make_path = os.path.join(dataset_path, make)
            # print(make)
            if os.path.isdir(make_path):
                for model in os.listdir(make_path):
                    model_path = os.path.join(make_path, model)
                    if os.path.isdir(model_path):
                        for generation in os.listdir(model_path):
                            generation_path = os.path.join(model_path, generation)
                            if os.path.isdir(generation_path):
                                for image_file in os.listdir(generation_path):
                                    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                        image_path = os.path.join(generation_path, image_file)
                                        images.append(image_path)
                                        labels.append(generation)
                                        # print("make = ", make)
                                        # print("model = ", model)
                                        # print("generation = ", generation)

        return images, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
                # Resize the image
        new_size = (224, 224)  # Replace with your desired size
        image = image.resize(new_size, Image.Resampling.LANCZOS)

        label = self.get_label(img_path)
        
        if self.transform:
            image = self.transform(image)
        # self.imshow(image)
        # plt.show()
        return image, label

    def get_label(self, img_path):
        # Extract label from the image path
        # You may need to adjust this based on your dataset structure
        parts = img_path.split(os.path.sep)
        make, model, generation, filename = parts[-4], parts[-3], parts[-2], parts[-1]
        label = generation
        # print(self.CLASSES.index(label))
        # print(len((self.CLASSES)))
        return self.CLASSES.index(label)

    def imshow(self, img):
        print("showing images")
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


class Solver(CarDataset, object):
    def __init__(self, config, dataset_path):
        super(Solver, self).__init__(root_dir=dataset_path)

        # Access CLASSES variable
        # self.CLASSES
        self.model = None
        self.lr = config.lr
        self.epochs = config.epoch
        self.train_batch_size = config.trainBatchSize
        self.val_batch_size = config.valBatchSize 
        self.test_batch_size = config.testBatchSize
        self.total_batch_size = config.BatchSize
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.Temp = 10
        self.car_dataset_root = dataset_path
        self.train_losses = []  # List to store training losses at each epoch
        self.val_losses = []    # List to store validation losses at each epoch
        self.accuracies = []    # List to store validation accuracies at each epoch

    def load_data(self):
        """
            load the image data from CIFAR-10
        """
        transform = transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

        ])

        # Create the CarDataset
        car_dataset = CarDataset(root_dir=self.car_dataset_root, transform=transform)

        # Define the sizes for train, validation, and test sets
        total_size = len(car_dataset)
        train_size = int(0.8 * total_size)
        val_size = (total_size - train_size) // 2
        test_size = total_size - train_size - val_size

        # Split the dataset into train, validation, and test sets
        train_data, val_data, test_data = torch.utils.data.random_split(car_dataset, [train_size, val_size, test_size])

        # Create DataLoader for each set
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.train_batch_size, shuffle=True, num_workers=4)
        self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.val_batch_size, shuffle=False, num_workers=4)
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.test_batch_size, shuffle=False, num_workers=4)


        # train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        # test_transform = transforms.Compose([transforms.ToTensor()])
        # # Create train and test datasets
        # train_set = CarDataset(root_dir=self.car_dataset_root, transform=train_transform)
        # test_set = CarDataset(root_dir=self.car_dataset_root, transform=test_transform)

        # # Use DataLoader with the datasets
        # self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.train_batch_size, shuffle=True)
        # self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size, shuffle=False)

        # train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        # self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.train_batch_size, shuffle=True)
        # test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        # self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size, shuffle=False)

    def load_model(self):
        """
            load GoogLeNet model or load pretrained model 
        """
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
            print("using cuda")
        else:
            self.device = torch.device('cpu')

        ## Uncommand if one wish to retrain the network
        # self.model = GoogLeNet(193).to(self.device)
        # self.model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True).to(self.device)
        # self.model = model.to(self.device)
        # self.model = models.googlenet(pretrained=True, num_classes=len(self.CLASSES))
        # # Modify the final fully connected layer

        # ## Load previously trained model
        # # self.model = torch.load('model_car_classification.pth')

        # self.model = torch.load('model_1.pth', map_location=torch.device('cpu')) 
        #self.model = torch.load('model_car1.pth').to(self.device)

        self.model = make_model('googlenet', num_classes=len(self.CLASSES), pretrained=True, input_size=(224, 224))
        # in_features = self.model.fc.in_features  # Get the number of input features
        # num_classes = len(self.CLASSES)  # Replace with the number of classes in your dataset
        # self.model.fc = torch.nn.Linear(in_features, num_classes)
        model_weight_path = './saved_models/model_epoch_4.pth'
        self.model.load_state_dict(torch.load(model_weight_path))
        self.model = self.model.to(self.device)
        print("car model loaded")
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75, 150], gamma=0.5)

        self.model_defense = make_model('googlenet', num_classes=len(self.CLASSES), pretrained=True, input_size=(224, 224))
        model_defense_weight_path = './saved_models/defense_epoch_100.pth'
        self.model_defense.load_state_dict(torch.load(model_defense_weight_path))
        self.model_defense = self.model_defense.to(self.device)
        print("defense model loaded")
        self.optimizer_d = optim.Adam(self.model_defense.parameters(), lr=self.lr)
        self.scheduler_d = optim.lr_scheduler.MultiStepLR(self.optimizer_d, milestones=[75, 150], gamma=0.5)

        self.criterion = nn.CrossEntropyLoss().to(self.device)



# ===================================================== Training, Validatoin, Tesing ===============================================================================

    def train(self):
        """
            train the GoogLeNet model 
        """
        print("train:")
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            # print("output ", output, "target ", target)
            # print("output size ", (output.shape), "data size ", (data.shape))
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total += target.size(0)

            # train_correct incremented by one if predicted corrected
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

            progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_num + 1), 100. * train_correct / total, train_correct, total))

        # Store training loss for plotting
        # self.train_losses.append(train_loss / len(self.train_loader))
        return train_loss, train_correct / total

    def validate(self):
        """
        validate the GoogLeNet model 
        """
        print("validate:")
        self.model.eval()
        val_loss = 0
        val_correct = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.val_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                val_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)
                val_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

        accuracy = val_correct / total
        # Store validation loss for plotting
        self.val_losses.append(val_loss / len(self.val_loader))
        return val_loss, accuracy

    def test(self):
        """
            test function using the testing set from the CIFAR-10
            display the accuracy for each class of objects
        """
        print("test:")
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0

        with torch.no_grad():           
            # prepare to count predictions for each classes
            correct_pred = {classname: 0 for classname in self.CLASSES}
            total_pred = {classname: 0 for classname in self.CLASSES} 

            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())


            # test different classes
                for label, predict in zip(target, prediction[1]):
                    # print(label)
                    # if label >= len(self.CLASSES):
                    #     print(f"Target value {label} is out of bounds for CLASSES list.")
                    # else:
                    if label == predict:
                        correct_pred[self.CLASSES[label]] += 1
                    total_pred[self.CLASSES[label]] += 1   

                progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))
        # print("Target values:", target)
 
        # accuracy for different classes
        # for classname, correct_count in correct_pred.items():
        #     accuracy = 100 * float(correct_count) / total_pred[classname]
        #     print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %, correct count: {correct_count}, total count: {total_pred[classname]}')

        return test_loss, test_correct / total

    def plot_losses(self):
        """
        Plot training and validation losses over epochs
        """
        plt.figure(figsize=(10, 50))
        plt.plot(self.train_losses, label='Training Loss')
        # plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses Over Epochs')
        plt.legend()
        plt.savefig('./result/loss_vs_epoch_defense.png')

        # plt.show()

    def plot_accuracy(self):
        """
        Plot training and validation losses over epochs
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.accuracies, label='Training Accuracy')
        # plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation using FGSM Attack (epsilon = 0.005) Accuracy Over Epochs')
        plt.legend()
        plt.savefig('./result/accuracy_vs_epoch_defense.png')

        # plt.show()


# ===================================================== FGSM Attack ===============================================================================

    def fgsm_attack(self, x, eps, data_grad):
        """
            fgsm attack -> Create the perturbed image by adjusting each pixel of the input image
                        based on the element-wise sign of the data gradient

            x: torch.Tensor (The input image)
            eps: noise magnitude
            data_grad: the gradient of the input data
        """

        pert_out = x + eps * data_grad.sign()
        pert_out = torch.clamp(pert_out, 0, 1)           # Adding clipping to maintain [0,1] range

        return pert_out


# ===================================================== noise attack ===============================================================================

    def noise_attack(self, x, eps):
        """
            noise attack -> create a tensor of the same shape as the input image x
                            and then make it a uniform distribution between -eps and  +eps

            x: torch.Tensor (The input image)
            eps: noise magnitude
        """
        eta = torch.FloatTensor(*x.shape).uniform_(-eps, eps).to(self.device)
        adv_x = x + eta

        return adv_x



# ===================================================== semantic attack ===============================================================================

    def semantic_attack(self, x):
        """
            semantic attack -> returns the negated image using the max value subtracting the image
            x: torch.Tensor (The input image)
        """
        return torch.max(x) - x
        


# ===================================================== show attack images ===============================================================================

    def img_attack_show(self):
        # get some random training images
        self.load_data()
        self.load_model()

        dataiter = iter(self.train_loader)
        data, labels = next(dataiter)
        data = data.to(self.device)
        data = data[0:4]
        # create attacked images
        epsilon = 0.3
        noise_img = self.noise_attack(data, epsilon)
        semantic_img = self.semantic_attack(data)

        rows = 2
        columns = 2
        # show images
        grid = torchvision.utils.make_grid(data[0:4])
        grid = grid / 2 + 0.5     # unnormalize
        npimg = grid.cpu().numpy()

        fig = plt.figure()
        fig.suptitle(r'Original Images and Adversarial Examples', fontsize=18, y=0.95)
        fig.add_subplot(rows, columns, 1)
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')
        plt.title(r"Original Image from Car Dataset")

        grid = torchvision.utils.make_grid(noise_img)
        grid = grid / 2 + 0.5     # unnormalize
        npimg = grid.cpu().numpy()
        fig.add_subplot(rows, columns, 2)
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')
        plt.title(r'Image under Noise Attack ($\varepsilon$ = {})'.format(epsilon))

        grid = torchvision.utils.make_grid(semantic_img)
        grid = grid / 2 + 0.5     # unnormalize
        npimg = grid.cpu().numpy()
        fig.add_subplot(rows, columns, 3)

        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')
        plt.title(r"Image under Semantic Attack")

        # plt.savefig("./result/adversarial_examples.png")
        # plt.show()

 
        self.model.eval()
        for data, target in self.train_loader:

            data, target = data.to(self.device), target.to(self.device)
            data.requires_grad = True
            output = self.model(data)

            loss = self.criterion(output, target)
            self.model.zero_grad()
            loss.backward()
            data_grad = data.grad.data

            # Call FGSM Attack
            perturbed_data = self.fgsm_attack(data, epsilon, data_grad)

        grid = torchvision.utils.make_grid(perturbed_data[0:4])
        grid = grid / 2 + 0.5     # unnormalize
        npimg = grid.cpu().numpy()
        fig.add_subplot(rows, columns, 4)
        mng = plt.get_current_fig_manager()
                # mng.resize(*mng.window.maxsize())
        mng.resize(1000, 600)
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')
        plt.title(r'Image under FGSM Attack ($\varepsilon$ = {})'.format(epsilon))
        plt.savefig("./result/adversarial_examples.png")
        plt.show()



# ===================================================== defense ===============================================================================

    def defense(self):
        """
            implementing the defensive distillation to counter the FGSM attack

            defensive distillation: using the output of the originally trained model
                                    to train another Neural Network model
        
        """
        # creating a new model and its optimizer and scheduler
        # self.load_data()
        # self.load_model()
        modelF1 = make_model('googlenet', num_classes=len(self.CLASSES), pretrained=True, input_size=(224, 224)).to(self.device)
        # model_defense_weight_path = './saved_models/defense_epoch_30.pth'
        modelF1.load_state_dict(torch.load(model_defense_weight_path))

        # modelF1 = make_model('googlenet', num_classes=len(self.CLASSES), pretrained=True, input_size=(224, 224)).to(self.device)
        optimizerF1 = optim.Adam(modelF1.parameters(), lr=self.lr)
        schedulerF1 = optim.lr_scheduler.MultiStepLR(optimizerF1, milestones=[75, 150], gamma=0.5)

        # train distilled Network
        accuracy = 0
        best_accuracy = 0

        F1_epoch = 100
        for epoch in range(1, F1_epoch + 1):
            schedulerF1.step(epoch)
            print("\n===> epoch: %d/%d" % (epoch, F1_epoch))
            train_result = self.train_F1(modelF1, optimizerF1)
            print(f"Training Results - Loss: {train_result[0]:.4f}, Accuracy: {train_result[1] * 100:.3f}%")
            # test_result = self.test_attack_F1(modelF1, 0.1, "fgsm", cnt)
            # print(f"Testing Results - Loss: {test_result[0]:.4f}, Accuracy: {test_result[1] * 100:.3f}%")
                        # Validation
            # val_result = self.test_attack_defense(modelF1, 0.0, "fgsm")
            val1_result = self.test_attack_defense(modelF1, 0.005, "fgsm")
            self.accuracies.append(val1_result[0])
            if(epoch % 5 == 0):
                model_path = f'./saved_models/defense_epoch_{epoch}.pth'
                torch.save(modelF1.state_dict(), model_path)
                print(f'Model saved at epoch {epoch} to {model_path}')

        self.plot_losses()
        self.plot_accuracy()

    def warmup_defense(self):
        # self.load_data()
        # self.load_model()
        accuracy = 0
        best_accuracy = 0

        F1_epoch = 3
        for epoch in range(1, F1_epoch + 1):
            self.scheduler_d.step(epoch)
            print("\n===> epoch: %d/%d" % (epoch, F1_epoch))
            train_result = self.train_F1(self.model_defense, self.optimizer_d)
            print(f"Training Results - Loss: {train_result[0]:.4f}, Accuracy: {train_result[1] * 100:.3f}%")
            # test_result = self.test_attack_F1(modelF1, 0.1, "fgsm", cnt)
            # print(f"Testing Results - Loss: {test_result[0]:.4f}, Accuracy: {test_result[1] * 100:.3f}%")
                        # Validation
            val_result = self.test_attack_defense(self.model_defense, 0.0, "fgsm")
            val1_result = self.test_attack_defense(self.model_defense, 0.005, "fgsm")


    # def train_F1(self, model, optimizer):
    #     """
    #         training the new defense network model
    #     """

    #     print("train:")
    #     model.train()
    #     train_loss = 0
    #     train_correct = 0
    #     total = 0

    #     for batch_num, (data, target) in enumerate(self.train_loader):
    #         data, target = data.to(self.device), target.to(self.device)             
    #         output = self.model(data)
    #         output = output/self.Temp
    #         optimizer.zero_grad()
    #         loss = self.criterion(output, target)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         train_loss += loss.item()
    #         prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
    #         total += target.size(0)

    #         # train_correct incremented by one if predicted correctly
    #         train_correct += np.sum(prediction[1].detach().cpu().numpy() == target.detach().cpu().numpy())

    #         progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
    #                      % (train_loss / (batch_num + 1), 100. * train_correct / total, train_correct, total))

    #     return train_loss/len(self.train_loader), train_correct / total

    def train_F1(self, model, optimizer):
        """
        training the new defense network model
        """

        print("train:")
        model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            softened_output = F.softmax(output / self.Temp, dim=1)  # Softmax with temperature
            optimizer.zero_grad()
            loss = F.kl_div(F.log_softmax(model(data) / self.Temp, dim=1), softened_output, reduction='batchmean')
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            prediction = torch.max(output, 1)
            total += target.size(0)
            train_correct += np.sum(prediction[1].detach().cpu().numpy() == target.detach().cpu().numpy())

            progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                        % (train_loss / (batch_num + 1), 100. * train_correct / total, train_correct, total))
        # Store training loss for plotting
        self.train_losses.append(train_loss / len(self.train_loader))
        return train_loss, train_correct / total

# ===================================================== test attacks ===============================================================================

    def run_attack(self):
        """
            evaluate the model with 3 different attacks
            for FGSM, a denfensive distillation model was used to deal with the attack
        """ 
        # self.load_data()
        # self.load_model()
        # can change the epsilons to desired values
        # epsilons = [0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
        epsilons = [0.0, 0.3]


        ## run through all of the attcks
        for attack in ("noise", "semantic"):
        # for attack in ("fgsm", "semantic"):
            print(attack)
            accuracies = []
            accuracies_d = []
            examples = []

            plt.figure()
            if attack == "fgsm":
                 # test attack on model without defense
                print("\nresults for original GoogleNet:")
                plt.subplots_adjust(hspace=0.5)
                plt.suptitle(r'Accuracy for Each Class (without defense) under FGSM Attack', fontsize=18, y=0.95)
                mng = plt.get_current_fig_manager()
                # mng.resize(*mng.window.maxsize())
                mng.resize(1000, 600)
                cnt = 0
                for eps in epsilons:
                    acc, ex = self.test_attack(eps, attack, cnt)      
                    accuracies.append(acc)
                    cnt+=1

                #  # test attack on defense network
                print("\nresults for GoogleNet with defense:")

                plt.subplots_adjust(hspace=0.5)
                plt.suptitle(r'Accuracy for Each Class (with defense) under FGSM Attack', fontsize=18, y=0.95)
                mng = plt.get_current_fig_manager()
                # mng.resize(*mng.window.maxsize())
                mng.resize(1000, 600)
                cnt = 0
                for eps in epsilons:
                    acc, ex = self.test_attack_F1(eps, attack, cnt)     
                    accuracies_d.append(acc)
                    cnt+=1


            
                # plot the results

                plt.plot(epsilons, accuracies, "*-", label = r'GoogleNet w/o defense')
                plt.plot(epsilons, accuracies_d, "*-", label = r'GoogleNet w/ defensive-distillation')
            
                plt.title(r"FGSM Attack Accuracy Comparison with and without Defense")
                plt.legend(loc='center right')
                plt.xlabel(r"Epsilon")
                plt.ylabel(r"Accuracy")
                plt.savefig('./result/fgsm_comparison.png')

                plt.show() 

            elif attack == "noise":

                cnt = 0
                for eps in epsilons:
                    acc, ex = self.test_attack(eps, attack, cnt)       # test attack on regular NN
                    accuracies.append(acc)
                    cnt+=1

                plt.figure()
                plt.subplots_adjust(hspace=0.5)
                plt.suptitle(r'Accuracy for Each Class (without defense) under Noise Attack', fontsize=18, y=0.95)
                mng = plt.get_current_fig_manager()
                mng.resize(1000, 600)
                plt.plot(epsilons, accuracies, "*-", label = r'GoogleNet w/o defense')
                plt.title(r"Accuracy for Each Class (without defense) under Noise Attack")
                plt.legend(loc='center right')
                plt.xlabel(r"Epsilon")
                plt.ylabel(r"Accuracy")
                plt.savefig('./result/class_acc_noise.png')
                
                # plt.show() 

            elif attack == "semantic":

                cnt = 0
                acc, ex = self.test_attack(0, attack, cnt)       # test attack on regular NN
                accuracies.append(acc)
                cnt+=1






    def test_attack(self, epsilon, attack, cnt):
        """
            test attack on the model
        """
        self.model.eval()
        correct = 0
        adv_examples = []
        correct_pred = {classname: 0 for classname in self.CLASSES}
        total_pred = {classname: 0 for classname in self.CLASSES} 
        for num, (data, target) in enumerate(self.test_loader):
            data, target = data.to(self.device), target.to(self.device)
            data.requires_grad = True
            output = self.model(data)

            init_pred = output.max(1, keepdim=True)[1] 
            # if init_pred.item() != target.item():
            # if not torch.equal(init_pred, target):
            #     continue
            loss = self.criterion(output, target)
            self.model.zero_grad()
            loss.backward()
            data_grad = data.grad.data

            if attack == "fgsm":
                perturbed_data = self.fgsm_attack(data, epsilon, data_grad)
            elif attack == "noise":
                perturbed_data = self.noise_attack(data, epsilon)
            elif attack == "semantic":
                perturbed_data = self.semantic_attack(data)
        
            output = self.model(perturbed_data)         
            # prediction = torch.max(output, 1)

            final_pred = output.max(1, keepdim=True)[1]
            if final_pred.item() == target.item():
                # print("correct prediction")
                correct += 1
                if (epsilon == 0) and (len(adv_examples) < 5):
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
                    # print(f"Adversarial example added for epsilon={epsilon}, len(adv_examples)={len(adv_examples)}")

            else:
                # print("wrong prediction")
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
                    # print(f"Adversarial example added for epsilon={epsilon}, len(adv_examples)={len(adv_examples)}")

            # test different class
            for label, predict in zip(target, final_pred):
                if label == predict:
                    correct_pred[self.CLASSES[label]] += 1
                total_pred[self.CLASSES[label]] += 1    

        final_acc = correct/float(len(self.test_loader))
        if attack == "fgsm" or attack == "noise":
            print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(self.test_loader), final_acc))
            x, y = zip(*correct_pred.items())
            # ax = plt.subplot(3, 3, cnt + 1)
            # # print(len())
            # # cmap = cm.jet(np.linspace(0, 1, len(correct_pred[0])))
            # ax.bar(np.arange(len(x)), y)
            # ax.set_xticks(np.arange(len(x)), x)
            # ax.set_title(r'$\varepsilon$ = {} (Total accuracy = {})'.format(epsilon, final_acc))
            # ax.set_ylim([0, 1100])
            # ax.set_xlabel(r"Class")
            # ax.set_ylabel(r"Accuracy")
            # plt.show(block=False)
        else:
            print("Semantic: \tTest Accuracy = {} / {} = {}".format(correct, len(self.test_loader), final_acc))
            x, y = zip(*correct_pred.items())
            # print(len())
            # cmap = cm.jet(np.linspace(0, 1, len(correct_pred[0])))
            # plt.bar(np.arange(len(x)), y)
            # plt.xticks(np.arange(len(x)), x)
            # plt.title(r"Accuracy for Each Class (without defense) under Semantic Attack (Total accuracy = {})".format(final_acc), fontsize=18, y=0.95)
            # plt.ylim([0, 1100])
            # plt.xlabel(r"Class")
            # plt.ylabel(r"Accuracy")
            # plt.show(block=False)
        
        # accuracy for different class 
        # for classname, correct_count in correct_pred.items():
        #     accuracy = 0
        #     if total_pred[classname] != 0:
        #         accuracy = 100 * float(correct_count) / total_pred[classname]
        #     print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

        return final_acc, adv_examples


    def test_attack_F1(self, epsilon, attack, cnt):
        """
            test attack on the defense model
        """
        self.model_defense.eval()
        correct = 0
        adv_examples = []
        correct_pred = {classname: 0 for classname in self.CLASSES}
        total_pred = {classname: 0 for classname in self.CLASSES} 
        for num, (data, target) in enumerate(self.test_loader):
            data, target = data.to(self.device), target.to(self.device)
            data.requires_grad = True
            output = self.model_defense(data)

            init_pred = output.max(1, keepdim=True)[1] 
            if init_pred.item() != target.item():
                continue
            loss = self.criterion(output, target)
            self.model_defense.zero_grad()
            loss.backward()
            data_grad = data.grad.data

            if attack == "fgsm":
                perturbed_data = self.fgsm_attack(data, epsilon, data_grad)
            elif attack == "noise":
                perturbed_data = self.noise_attack(data, epsilon)
            elif attack == "semantic":
                perturbed_data = self.semantic_attack(data)
        
            output = self.model_defense(perturbed_data)
            # prediction = torch.max(output, 1)
            final_pred = output.max(1, keepdim=True)[1]
            if final_pred.item() == target.item():
                correct += 1
                if (epsilon == 0) and (len(adv_examples) < 5):
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
            else:
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

            # test different class
            for label, predict in zip(target, final_pred):
                if label == predict:
                    correct_pred[self.CLASSES[label]] += 1
                total_pred[self.CLASSES[label]] += 1   

        final_acc = correct/float(len(self.test_loader))
        if attack == "fgsm" or attack == "noise":
            print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(self.test_loader), final_acc))
        else:
            print("semantic: \tTest Accuracy = {} / {} = {}".format(correct, len(self.test_loader), final_acc))

        # accuracy for different class 
        # for classname, correct_count in correct_pred.items():
        #     accuracy = 100 * float(correct_count) / total_pred[classname]
        #     print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

        # x, y = zip(*correct_pred.items())
        # ax = plt.subplot(3, 3, cnt + 1)
        # # print(len())
        # # cmap = cm.jet(np.linspace(0, 1, len(correct_pred[0])))
        # ax.bar(np.arange(len(x)), y)
        # ax.set_xticks(np.arange(len(x)), x)
        # ax.set_title(r'$\varepsilon$ = {} (Total accuracy = {})'.format(epsilon, final_acc))
        # ax.set_ylim([0, 1100])
        # ax.set_xlabel(r"Class")
        # ax.set_ylabel(r"Accuracy")

        return final_acc, adv_examples


    def test_attack_defense(self, model, epsilon, attack):
        """
            test attack on the defense model
        """
        model.eval()
        correct = 0
        adv_examples = []
        correct_pred = {classname: 0 for classname in self.CLASSES}
        total_pred = {classname: 0 for classname in self.CLASSES} 
        for num, (data, target) in enumerate(self.test_loader):
            data, target = data.to(self.device), target.to(self.device)
            data.requires_grad = True
            output = model(data)

            init_pred = output.max(1, keepdim=True)[1] 
            if init_pred.item() != target.item():
                continue
            loss = self.criterion(output, target)
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data

            if attack == "fgsm":
                perturbed_data = self.fgsm_attack(data, epsilon, data_grad)
            elif attack == "noise":
                perturbed_data = self.noise_attack(data, epsilon)
            elif attack == "semantic":
                perturbed_data = self.semantic_attack(data)
        
            output = model(perturbed_data)
            # prediction = torch.max(output, 1)
            final_pred = output.max(1, keepdim=True)[1]
            if final_pred.item() == target.item():
                correct += 1
                if (epsilon == 0) and (len(adv_examples) < 5):
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
            else:
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

            # test different class
            for label, predict in zip(target, final_pred):
                if label == predict:
                    correct_pred[self.CLASSES[label]] += 1
                total_pred[self.CLASSES[label]] += 1   

        final_acc = correct/float(len(self.test_loader))
        if attack == "fgsm" or attack == "noise":
            print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(self.test_loader), final_acc))
        else:
            print("semantic: \tTest Accuracy = {} / {} = {}".format(correct, len(self.test_loader), final_acc))

        return final_acc, adv_examples

# ======================================================= save and run =======================================================================================

    def save(self):
        """
            save the trained model to designated path
        """
        model_out_path = "model_car_classification.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def run(self):
        """
            train the model and run the trained model with the regular testing function 
        """
        self.load_data()
        self.load_model()
        accuracy = 0
        best_accuracy = 0

        for epoch in range(1, self.epochs + 1):
            self.scheduler.step(epoch)
            print("\n===> epoch: %d/%d" % (epoch, self.epochs))

            # Training
            train_result = self.train()
            print(f"Training Results - Loss: {train_result[0]:.4f}, Accuracy: {train_result[1] * 100:.3f}%")

            # Validation
            val_result = self.validate()
            print(f"Validation Results - Loss: {val_result[0]:.4f}, Accuracy: {val_result[1] * 100:.3f}%")

            # Update best accuracy
            best_accuracy = max(best_accuracy, val_result[1])

            # Save the model after each epoch
            model_path = f'./saved_models/model_epoch_{epoch+1}.pth'
            torch.save(self.model.state_dict(), model_path)
            print(f'Model saved at epoch {epoch+1} to {model_path}')

        print(f"\nBest Validation Accuracy: {best_accuracy * 100:.3f}%")

        # Testing
        test_result = self.test()
        accuracy = max(accuracy, test_result[1])
        print(test_result[0], accuracy)

        # self.save()
        # self.plot_losses()

        # if epoch == self.epochs:
        #     print("===> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))

    def warm_up(self):
        """
            train the model and run the trained model with the regular testing function 
        """
        self.load_data()
        self.load_model()
        accuracy = 0
        best_accuracy = 0
        warm_up_epochs = 3
        for epoch in range(1, warm_up_epochs + 1):
            self.scheduler.step(epoch)
            print("\n===> epoch: %d/%d" % (epoch, warm_up_epochs))

            # Training
            train_result = self.train()
            print(f"Training Results - Loss: {train_result[0]:.4f}, Accuracy: {train_result[1] * 100:.3f}%")

            # Validation
            val_result = self.validate()
            print(f"Validation Results - Loss: {val_result[0]:.4f}, Accuracy: {val_result[1] * 100:.3f}%")

            # Update best accuracy
            best_accuracy = max(best_accuracy, val_result[1])

        print(f"\nBest Validation Accuracy: {best_accuracy * 100:.3f}%")

        # Testing
        test_result = self.test()
        accuracy = max(accuracy, test_result[1])
        print(test_result[0], accuracy)


def main():
    parser = argparse.ArgumentParser(description="Car Model Classification with PyTorch")
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epoch', default=3, type=int, help='number of epochs to train for')
    ## original testing -> test size = 100 (uncomment this line to show regular testing results)
    # parser.add_argument('--trainBatchSize', default=100, type=int, help='training batch size')
    # parser.add_argument('--testBatchSize', default=100, type=int, help='testing batch size')
    
    ## run attack -> test size = 1 (uncomment this line to show attack results)
    parser.add_argument('--trainBatchSize', default=50, type=int, help='training batch size')
    parser.add_argument('--valBatchSize', default=10, type=int, help='testing batch size')
    parser.add_argument('--testBatchSize', default=1, type=int, help='testing batch size')
    parser.add_argument('--BatchSize', default=100, type=int, help='testing batch size')

    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    args = parser.parse_args()

    ## Solver Section
    dataset_path = './car-dataset-200/riotu-cars-dataset-200'

    # car_data = CarDataset(dataset_path)
    solver = Solver(args, dataset_path)
    # solver.run()
    solver.warm_up()
    # solver.warmup_defense()
    # solver.img_attack_show()
    solver.run_attack()
    # solver.defense()
    # solver.test_distillation()

if __name__ == '__main__':
    main()

