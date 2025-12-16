import torch





def main():
    print("Hello from breeze-asr-25!")
    print(f"CUDA is available: {torch.cuda.is_available()}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA current device: {torch.cuda.current_device()}")
    print(f"CUDA device properties: {torch.cuda.get_device_properties(0)}")
    print(f"CUDA device properties: {torch.cuda.get_device_properties(0)}")


if __name__ == "__main__":
    main()
