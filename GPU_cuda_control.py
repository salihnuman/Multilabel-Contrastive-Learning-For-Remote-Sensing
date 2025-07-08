import torch

def set_device():
    if torch.cuda.is_available() == True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(device)

def print_device_info():
    
    print("There are " + str(torch.cuda.device_count()) + " device(s) on this system.")
    print("The current device is " + str(torch.cuda.get_device_name(torch.cuda.current_device())))

    print("Pytorch version is " + torch.__version__)
    print("Cuda version is " + torch.version.cuda)

# print(os.getcwd()

def main():
    set_device()
    print_device_info()



if __name__ == "__main__":
    main()
