import argparse
import os
import subprocess
import sys
import torch
import torch.distributed as dist
import time

def parse_size(size_str):
    """Parse memory size string like '128M' or '1G' into bytes."""
    units = {'K': 1024, 'M': 1024**2, 'G': 1024**3}
    if size_str[-1].upper() not in units:
        raise ValueError(f"Invalid size unit in '{size_str}'. Expected one of {list(units.keys())}.")
    unit = size_str[-1].upper()
    number = float(size_str[:-1])
    return int(number * units[unit])

def setup(backend, gpus_per_node):
    """Initialize distributed environment based on environment variables."""
    if not gpus_per_node:
        raise ValueError("GPU count per node is required.")
    dist.init_process_group(
        backend=backend,
    )
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if backend == 'nccl':
        torch.cuda.set_device(rank % torch.cuda.device_count())

def allgather_run(cmd):
    """Execute a command and gather outputs across all processes."""
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = result.stdout.decode('utf-8')
    outputs = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(outputs, output)
    return outputs

def allequal(iterator):
    """Check if all process outputs are identical."""
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)

def benchmark_nccl_communication(begin_size, end_size, factor, gpus_node, num_tests=10):
    """Conduct NCCL communication performance tests."""
    print(f"NCCL communication benchmark from {begin_size} to {end_size} by {factor}x factor.")
    size = parse_size(begin_size)
    end_size = parse_size(end_size)
    while size <= end_size:
        
        num_elements = size // (torch.finfo(torch.float32).bits // 8)


        tensor = torch.rand(num_elements, device='cuda')
        tensor_size = tensor.element_size()
        print(f"Tensor size: {tensor_size} bytes")
        dist.barrier()
        start_time = time.time()
        
        dist.all_reduce(tensor)
        dist.barrier()

        duration = (time.time() - start_time)
        algbw = (size / duration) / 1e9
        busbw = algbw * (2 * (gpus_node - 1) / gpus_node)
        print(f"Size: {size} bytes, Duration: {duration:.6f}s, Algbw: {algbw:.2f} GB/s, Busbw: {busbw:.2f} GB/s")
        size *= factor

        
def main():
    parser = argparse.ArgumentParser(description="Simulate nccl-test for distributed communication performance testing.")
    parser.add_argument("-b", "--begin-size", default="8", help="Starting data size.")
    parser.add_argument("-e", "--end-size", default="128M", help="Ending data size.")
    parser.add_argument("-f", "--factor", type=int, default=2, help="Data size growth factor.")
    parser.add_argument("-g", "--gpus", type=int, required=True, help="GPUs per node (ignored if using mpirun).")
    args = parser.parse_args()

    try:
        torch_version = torch.__version__
        print(f"Checking PyTorch compatibility: ...........  v{torch_version} [\033[32mPASSED\033[0m]")
    except Exception as e:
        print("Checking PyTorch compatibility: ...........  [\033[31mFAILED\033[0m]")
        return

    # Check CUDA availability if PyTorch compatibility is passed
    if torch.cuda.is_available():
        print("CUDA is available: ...........  [\033[32mPASSED\033[0m]")
    else:
        print("CUDA is available: ...........  [\033[31mFAILED\033[0m]")
        return

    # Check CUDA version supported by the PyTorch version
    try:
        cuda_version = torch.version.cuda
        print(f"CUDA version supported by PyTorch: ...........  {cuda_version} [\033[32mPASSED\033[0m]")
    except Exception as e:
        print("CUDA version supported by PyTorch: ...........  [\033[31mFAILED\033[0m]")
        return


    # Check number of available GPUs
    try:
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: ...........  {num_gpus} [\033[32mPASSED\033[0m]")
    except Exception as e:
        print("Number of available GPUs: ...........  [\033[31mFAILED\033[0m]")
        return

    # Check GPU type
    try:
        gpu_name = torch.cuda.get_device_name()
        print(f"GPU type: ...........  {gpu_name} [\033[32mPASSED\033[0m]")
    except Exception as e:
        print("GPU type: ...........  [\033[31mFAILED\033[0m]")
        return

    # Check NCCL version
    try:
        nccl_version = torch.cuda.nccl.version()
        print(f"NCCL version: ...........  {nccl_version} [\033[32mPASSED\033[0m]")
    except Exception as e:
        print("NCCL version: ...........  [\033[31mFAILED\033[0m]")
        return

    setup(backend='nccl', gpus_per_node=args.gpus)
    print(f"Rank: {dist.get_rank()}, World size: {dist.get_world_size()}")
    print(f"Using GPUs: {torch.cuda.current_device()}")
    # print(args.gpus)
    # outputs = allgather_run("nvidia-smi topo -m")
    # if not allequal(outputs):
    #     print('Output of "nvidia-smi topo -m" differs between machines')
    #     sys.exit(1)

    if dist.get_rank() == 0:
        print("-----------------------------------")
        print("PyTorch Distributed Benchmark Suite")
        print("-----------------------------------")
        print(f"* PyTorch version: {torch.__version__}")
        print(f"* CUDA version: {torch.version.cuda}")
        # print("--- nvidia-smi topo -m ---")
        # print(outputs[0])
        print("--------------------------")

    benchmark_nccl_communication(args.begin_size, args.end_size, args.factor,args.gpus)

if __name__ == "__main__":
    main()
