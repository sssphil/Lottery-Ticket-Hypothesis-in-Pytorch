# code
## main
First, set result directory and dump results by run date.
```python
def main(args, ITE=0)

    # dump results by date
    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    resultsdir = os.path.join(
        'results/' + current_time + '_' + socket.gethostname())
```
Use cuda and determine if reinit. This project probably won't use reinitialization.
```python
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reinit = True if args.prune_type=="reinit" else False
```
Initialize datasets. Add custom datasets here.
```python
    # Data Loader
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    if args.dataset == "mnist":
        traindataset = datasets.MNIST('../data', train=True, download=True,transform=transform)
        testdataset = datasets.MNIST('../data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet

    elif args.dataset == "cifar10":
        traindataset = datasets.CIFAR10('../data', train=True, download=True,transform=transform)
        testdataset = datasets.CIFAR10('../data', train=False, transform=transform)      
        from archs.cifar10 import AlexNet, LeNet5, fc1, vgg, resnet, densenet 

    elif args.dataset == "fashionmnist":
        traindataset = datasets.FashionMNIST('../data', train=True, download=True,transform=transform)
        testdataset = datasets.FashionMNIST('../data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet 

    elif args.dataset == "cifar100":
        traindataset = datasets.CIFAR100('../data', train=True, download=True,transform=transform)
        testdataset = datasets.CIFAR100('../data', train=False, transform=transform)   
        from archs.cifar100 import AlexNet, fc1, LeNet5, vgg, resnet  
    
    # If you want to add extra datasets paste here

    else:
        print("\nWrong Dataset choice \n")
        exit()
```
Set multiple workers and pin_memory to accelerate loading data.
```python
    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=8,
                                               pin_memory=True if device.type != 'cpu' else False,
                                               drop_last=False)
    #train_loader = cycle(train_loader)
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=8,
                                              pin_memory=True if device.type != 'cpu' else False,
                                              drop_last=True)
```
Choose model to train and prune. In this project, we use "fc1". The fully-connected network has two hidden layers (300, 100), the input layer has 3*32*32 neurons, and the output layer has 10 neurons. The activation function for all hidden layers is ReLU.
Hopefully we could make some improvements with [this paper](https://openreview.net/pdf/1WvovwjA7UMnPB1oinBL.pdf).
```python
    # Importing Network Architecture
    global model
    if args.arch_type == "fc1":
       model = fc1.fc1().to(device)
    elif args.arch_type == "lenet5":
        model = LeNet5.LeNet5().to(device)
    elif args.arch_type == "alexnet":
        model = AlexNet.AlexNet().to(device)
    elif args.arch_type == "vgg16":
        model = vgg.vgg16().to(device)  
    elif args.arch_type == "resnet18":
        model = resnet.resnet18().to(device)   
    elif args.arch_type == "densenet121":
        model = densenet.densenet121().to(device)
    # If you want to add extra model paste here
    else:
        print("\nWrong Model choice\n")
        exit()
```
Initialize model parameters. The original code uses Glorot initialization, but according to [this post](https://stats.stackexchange.com/questions/319323/whats-the-difference-between-variance-scaling-initializer-and-xavier-initialize), He initialization suits ReLU better. See documentation on [`torch.nn.module.apply(fn: Callable[Module, None])`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) and [`torch.nn.init`](https://pytorch.org/docs/stable/nn.init.html).
TODO: Initialization for the fully-connected network presented in the other reference.
```python
    # Weight Initialization
    model.apply(weight_init)
```
Save initial state of the model, weight and bias.
```python
    # Copying and Saving Initial State
    initial_state_dict = copy.deepcopy(model.state_dict())
    utils.checkdir(f"{os.getcwd()}/{resultsdir}/saves/{args.arch_type}/{args.dataset}/")
    torch.save(model, f"{os.getcwd()}/{resultsdir}/saves/{args.arch_type}/{args.dataset}/initial_state_dict_{args.prune_type}.pth.tar")
```
The method `make_mask(model)` makes an empty mask an empty mask of the same size as the model. The mask is a list variable, containing numpy arrays with the same size as weights per layer. Initially, all mask values are 1. 
```python
    # Making Initial Mask
    make_mask(model)
```
Adam algorithm and CrossEntropyLoss. Note CrossEntropyLoss contains a log softmax layer.
```
    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss() # Default was F.nll_loss
```
Some loop variables.
`bestacc`: list of best accuracy through epochs.
`all_loss`: list of training loss through epochs.
`all_accuracy`: list of testing accuracy through epochs. If not updated, the latest testing accuracy will be restored.
`comp`: list of unpruned weights percentage through epochs. 

```python
    # Layer Looper
    for name, param in model.named_parameters():
        print(name, param.size())

    # Pruning
    # NOTE First Pruning Iteration is of No Compression
    bestacc = 0.0
    best_accuracy = 0
    ITERATION = args.prune_iterations
    comp = np.zeros(ITERATION,float)
    bestacc = np.zeros(ITERATION,float)
    step = 0
    all_loss = np.zeros(args.end_iter,float)
    all_accuracy = np.zeros(args.end_iter,float)
```
### The main loop
Loop starts at `args.start_iter` and ends at `ITERATION`-1. `ITERATION` is set by `--prune_iterations`.
```python
    for _ite in range(args.start_iter, ITERATION):
```
The first iteration is for the full, unpruned model. For later iterations, the model will be pruned by percentage. It seems the prune function only uses the `prune_percent` argument (default 10).
The `original_initialization(mask_temp, initial_state_dict)` function applys the mask on the model weights. TODO: does the optimizer need to be reinitialized? The structure of the model remains unchanged, and `model.paramters()` returns the same weight and bias tensors but different values.
The pruning method will be discussed later.
```python    
        if not _ite == 0:
            prune_by_percentile(args.prune_percent, resample=resample, reinit=reinit)
            if reinit:
                model.apply(weight_init)
                #if args.arch_type == "fc1":
                #    model = fc1.fc1().to(device)
                #elif args.arch_type == "lenet5":
                #    model = LeNet5.LeNet5().to(device)
                #elif args.arch_type == "alexnet":
                #    model = AlexNet.AlexNet().to(device)
                #elif args.arch_type == "vgg16":
                #    model = vgg.vgg16().to(device)  
                #elif args.arch_type == "resnet18":
                #    model = resnet.resnet18().to(device)   
                #elif args.arch_type == "densenet121":
                #    model = densenet.densenet121().to(device)   
                #else:
                #    print("\nWrong Model choice\n")
                #    exit()
                step = 0
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        weight_dev = param.device
                        param.data = torch.from_numpy(param.data.cpu().numpy() * mask[step]).to(weight_dev)
                        step = step + 1
                step = 0
            else:
                original_initialization(mask, initial_state_dict)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
```
Print information. `tqdm` is used for progress bar and is also iterable. `args.end_iter` is the number of epochs.
```python
        print(f"\n--- Pruning Level [{ITE}:{_ite}/{ITERATION}]: ---")

        # Print the table of Nonzeros in each layer
        comp1 = utils.print_nonzeros(model)
        comp[_ite] = comp1
        pbar = tqdm(range(args.end_iter))
```
#### Loop for each epoch (inside main loop)
```python
        for iter_ in pbar:
```
Test the model every `args.valid_freq` epochs. By default, test the model during every epoch. First test happens after initialization before training. 
The model is saved using `torch.save` and can be loaded by `torch.load()`.
NOTE: `test(model, test_loader, criterion)` uses `F.nll_loss` instead of the `criterion` argument, and the loss is not returned.
```python
            # Frequency for Testing
            if iter_ % args.valid_freq == 0:
                accuracy = test(model, test_loader, criterion)

                # Save Weights
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    utils.checkdir(f"{os.getcwd()}/{resultsdir}/saves/{args.arch_type}/{args.dataset}/")
                    torch.save(model,f"{os.getcwd()}/{resultsdir}/saves/{args.arch_type}/{args.dataset}/{_ite}_model_{args.prune_type}.pth.tar")
```
Train the model with Adam and CrossEntropyLoss. The gradients of pruned weights are set to zero (these weights are "freezed" to 0) before `optimizer.step()`.
```python
            # Training
            loss = train(model, train_loader, optimizer, criterion)
            all_loss[iter_] = loss
            all_accuracy[iter_] = accuracy
            
            # Frequency for Printing Accuracy and Loss
            if iter_ % args.print_freq == 0:
                pbar.set_description(
                    f'Train Epoch: {iter_}/{args.end_iter} Loss: {loss:.6f} Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')       
```
#### plotting (after epoch loop, inside main loop)

```python
        writer.add_scalar('Accuracy/test', best_accuracy, comp1)
        bestacc[_ite]=best_accuracy

        # Plotting Loss (Training), Accuracy (Testing), Iteration Curve
        #NOTE Loss is computed for every iteration while Accuracy is computed only for every {args.valid_freq} iterations. Therefore Accuracy saved is constant during the uncomputed iterations.
        #NOTE Normalized the accuracy to [0,100] for ease of plotting.
        plt.plot(np.arange(1,(args.end_iter)+1), 100*(all_loss - np.min(all_loss))/np.ptp(all_loss).astype(float), c="blue", label="Loss") 
        plt.plot(np.arange(1,(args.end_iter)+1), all_accuracy, c="red", label="Accuracy") 
        plt.title(f"Loss Vs Accuracy Vs Iterations ({args.dataset},{args.arch_type})") 
        plt.xlabel("Iterations") 
        plt.ylabel("Loss and Accuracy") 
        plt.legend() 
        plt.grid(color="gray") 
        utils.checkdir(f"{os.getcwd()}/{resultsdir}/plots/lt/{args.arch_type}/{args.dataset}/")
        plt.savefig(f"{os.getcwd()}/{resultsdir}/plots/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_LossVsAccuracy_{comp1}.png", dpi=1200)
        plt.close()

        # Dump Plot values
        utils.checkdir(f"{os.getcwd()}/{resultsdir}/dumps/lt/{args.arch_type}/{args.dataset}/")
        all_loss.dump(f"{os.getcwd()}/{resultsdir}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_all_loss_{comp1}.dat")
        all_accuracy.dump(f"{os.getcwd()}/{resultsdir}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_all_accuracy_{comp1}.dat")
        
        # Dumping mask
        utils.checkdir(f"{os.getcwd()}/{resultsdir}/dumps/lt/{args.arch_type}/{args.dataset}/")
        with open(f"{os.getcwd()}/{resultsdir}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_mask_{comp1}.pkl", 'wb') as fp:
            pickle.dump(mask, fp)
        
        # Making variables into 0
        best_accuracy = 0
        all_loss = np.zeros(args.end_iter,float)
        all_accuracy = np.zeros(args.end_iter,float)
```
### dump results (after main loop)
Including `comp`, `bestacc`, and the accuracyVSweights figure.
```python
    # Dumping Values for Plotting
    utils.checkdir(f"{os.getcwd()}/{resultsdir}/dumps/lt/{args.arch_type}/{args.dataset}/")
    comp.dump(f"{os.getcwd()}/{resultsdir}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_compression.dat")
    bestacc.dump(f"{os.getcwd()}/{resultsdir}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_bestaccuracy.dat")

    # Plotting
    a = np.arange(args.prune_iterations)
    plt.plot(a, bestacc, c="blue", label="Winning tickets") 
    plt.title(f"Test Accuracy vs Unpruned Weights Percentage ({args.dataset},{args.arch_type})") 
    plt.xlabel("Unpruned Weights Percentage") 
    plt.ylabel("test accuracy") 
    plt.xticks(a, comp, rotation ="vertical") 
    plt.ylim(0,100)
    plt.legend() 
    plt.grid(color="gray") 
    utils.checkdir(f"{os.getcwd()}/{resultsdir}/plots/lt/{args.arch_type}/{args.dataset}/")
    plt.savefig(f"{os.getcwd()}/{resultsdir}/plots/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_AccuracyVsWeights.png", dpi=1200)
    plt.close()  
```

# Pruning method
Prune weights and update the pruning mask. The percent argument is passed by `args.prune_percent` (default 10).
Documentation: [`numpy.percentile`](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html) and [`numpy.where`](https://numpy.org/doc/stable/reference/generated/numpy.where.html)
```python
def prune_by_percentile(percent, resample=False, reinit=False,**kwargs):
        global step
        global mask
        global model

        # Calculate percentile value
        step = 0
        for name, param in model.named_parameters():

            # We do not prune bias term
            if 'weight' in name:
                tensor = param.data.cpu().numpy()
                alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
                percentile_value = np.percentile(abs(alive), percent)

                # Convert Tensors to numpy and calculate
                weight_dev = param.device
                new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])
                
                # Apply new weight and mask
                param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                mask[step] = new_mask
                step += 1
        step = 0

```
# results
1. Initial state
  - model -> {os.getcwd()}/{resultsdir}/saves/{args.arch_type}/{args.dataset}/initial_state_dict_{args.prune_type}.pth.tar
2. Weights
  - model -> {os.getcwd()}/{resultsdir}/saves/{args.arch_type}/{args.dataset}/{_ite}_model_{args.prune_type}.pth.tar
3. Figures
  - {os.getcwd()}/{resultsdir}/plots/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_LossVsAccuracy_{comp1}.png
4. Loss
  - {os.getcwd()}/{resultsdir}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_all_loss_{comp1}.dat
5. Accuracy
  - {os.getcwd()}/{resultsdir}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_all_accuracy_{comp1}.dat
6. Mask
  - {os.getcwd()}/{resultsdir}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_mask_{comp1}.pkl
7. Comp
  - {os.getcwd()}/{resultsdir}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_compression.dat
8. Best accuracy
  - {os.getcwd()}/{resultsdir}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_bestaccuracy.dat
9. Best accuracy VS Unpruned weights
  - {os.getcwd()}/{resultsdir}/plots/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_AccuracyVsWeights.png





# Lottery Ticket Hypothesis in Pytorch 
[![Made With python 3.7](https://img.shields.io/badge/Made%20with-Python%203.7-brightgreen)]() [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)]() [![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)]() 

This repository contains a **Pytorch** implementation of the paper [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635) by [Jonathan Frankle](https://github.com/jfrankle) and [Michael Carbin](https://people.csail.mit.edu/mcarbin/) that can be **easily adapted to any model/dataset**.
		
## Requirements
```
pip3 install -r requirements.txt
```
## How to run the code ? 
### Using datasets/architectures included with this repository :
```
python3 main.py --prune_type=lt --arch_type=fc1 --dataset=mnist --prune_percent=10 --prune_iterations=35
```
- `--prune_type` : Type of pruning  
	- Options : `lt` - Lottery Ticket Hypothesis, `reinit` - Random reinitialization
	- Default : `lt`
- `--arch_type`	 : Type of architecture
	- Options : `fc1` - Simple fully connected network, `lenet5` - LeNet5, `AlexNet` - AlexNet, `resnet18` - Resnet18, `vgg16` - VGG16 
	- Default : `fc1`
- `--dataset`	: Choice of dataset 
	- Options : `mnist`, `fashionmnist`, `cifar10`, `cifar100` 
	- Default : `mnist`
- `--prune_percent`	: Percentage of weight to be pruned after each cycle. 
	- Default : `10`
- `--prune_iterations`	: Number of cycle of pruning that should be done. 
	- Default : `35`
- `--lr`	: Learning rate 
	- Default : `1.2e-3`
- `--batch_size`	: Batch size 
	- Default : `60`
- `--end_iter`	: Number of Epochs 
	- Default : `100`
- `--print_freq`	: Frequency for printing accuracy and loss 
	- Default : `1`
- `--valid_freq`	: Frequency for Validation 
	- Default : `1`
- `--gpu`	: Decide Which GPU the program should use 
	- Default : `0`
### Using datasets/architectures that are not included with this repository :
- Adding a new architecture :
	- For example, if you want to add an architecture named `new_model` with `mnist` dataset compatibility. 
		- Go to `/archs/mnist/` directory and create a file `new_model.py`.
		- Now paste your **Pytorch compatible** model inside `new_model.py`.
		- **IMPORTANT** : Make sure the *input size*, *number of classes*, *number of channels*, *batch size* in your `new_model.py` matches with the corresponding dataset that you are adding (in this case, it is `mnist`).
		- Now open `main.py` and go to `line 36` and look for the comment `# Data Loader`. Now find your corresponding dataset (in this case, `mnist`) and add `new_model` at the end of the line `from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet`.
		- Now go to `line 82` and add the following to it :
			```
			elif args.arch_type == "new_model":
        		model = new_model.new_model_name().to(device)
			``` 
			Here, `new_model_name()` is the name of the model that you have given inside `new_model.py`.
- Adding a new dataset :
	- For example, if you want to add a dataset named `new_dataset` with `fc1` architecture compatibility.
		- Go to `/archs` and create a directory named `new_dataset`.
		- Now go to /archs/new_dataset/` and add a file named `fc1.py` or copy paste it from existing dataset folder.
		- **IMPORTANT** : Make sure the *input size*, *number of classes*, *number of channels*, *batch size* in your `new_model.py` matches with the corresponding dataset that you are adding (in this case, it is `new_dataset`).
		- Now open `main.py` and goto `line 58` and add the following to it :
			```
			elif args.dataset == "cifar100":
        		traindataset = datasets.new_dataset('../data', train=True, download=True, transform=transform)
        		testdataset = datasets.new_dataset('../data', train=False, transform=transform)from archs.new_dataset import fc1
			``` 
			**Note** that as of now, you can only add dataset that are [natively available in Pytorch](https://pytorch.org/docs/stable/torchvision/datasets.html). 

## How to combine the plots of various `prune_type` ?
- Go to `combine_plots.py` and add/remove the datasets/archs who's combined plot you want to generate (*Assuming that you have already executed the `main.py` code for those dataset/archs and produced the weights*).
- Run `python3 combine_plots.py`.
- Go to `/plots/lt/combined_plots/` to see the graphs.

Kindly [raise an issue](https://github.com/rahulvigneswaran/Lottery-Ticket-Hypothesis-in-Pytorch/issues) if you have any problem with the instructions. 


## Datasets and Architectures that were already tested

|              | fc1                | LeNet5                | AlexNet                | VGG16                | Resnet18                 |
|--------------|:------------------:|:---------------------:|:----------------------:|:--------------------:|:------------------------:|
| MNIST        | :heavy_check_mark: |  :heavy_check_mark:   |   :heavy_check_mark:   |  :heavy_check_mark:  |  	:heavy_check_mark:	   |
| CIFAR10      | :heavy_check_mark: |  :heavy_check_mark:   |   :heavy_check_mark:   |  :heavy_check_mark:  |	:heavy_check_mark:	   |
| FashionMNIST | :heavy_check_mark: |  :heavy_check_mark:   |   :heavy_check_mark:   |  :heavy_check_mark:  |	:heavy_check_mark:	   |
| CIFAR100     | :heavy_check_mark: |  :heavy_check_mark:   |   :heavy_check_mark:   |  :heavy_check_mark:  |	:heavy_check_mark:     |


## Repository Structure
```
Lottery-Ticket-Hypothesis-in-Pytorch
├── archs
│   ├── cifar10
│   │   ├── AlexNet.py
│   │   ├── densenet.py
│   │   ├── fc1.py
│   │   ├── LeNet5.py
│   │   ├── resnet.py
│   │   └── vgg.py
│   ├── cifar100
│   │   ├── AlexNet.py
│   │   ├── fc1.py
│   │   ├── LeNet5.py
│   │   ├── resnet.py
│   │   └── vgg.py
│   └── mnist
│       ├── AlexNet.py
│       ├── fc1.py
│       ├── LeNet5.py
│       ├── resnet.py
│       └── vgg.py
├── combine_plots.py
├── dumps
├── main.py
├── plots
├── README.md
├── requirements.txt
├── saves
└── utils.py

```

## Interesting papers that are related to Lottery Ticket Hypothesis which I enjoyed 
- [Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask](https://eng.uber.com/deconstructing-lottery-tickets/)

## Acknowledgement 
Parts of code were borrowed from [ktkth5](https://github.com/ktkth5/lottery-ticket-hyopothesis).

## Issue / Want to Contribute ? :
Open a new issue or do a pull request incase you are facing any difficulty with the code base or if you want to contribute to it.

[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://github.com/rahulvigneswaran/Lottery-Ticket-Hypothesis-in-Pytorch/issues)

<a href="https://www.buymeacoffee.com/MfgDK7aSC" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>

