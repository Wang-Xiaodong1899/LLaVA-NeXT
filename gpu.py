import torch
import multiprocessing as mp
import time
import sys

def gpu_worker(device_id):
    device = torch.device(f'cuda:{device_id}')

    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocate_memory = int(total_memory * 0.3)
    
    element_size = 4 
    num_elements = allocate_memory // element_size
    dim = int((num_elements // 2) ** 0.5)  
    

    a = torch.randn(dim, dim, device=device, dtype=torch.float32)
    b = torch.randn(dim, dim, device=device, dtype=torch.float32)
    
    with torch.no_grad():
        _ = a @ b
    torch.cuda.synchronize(device)
    

    duty_cycle = 0.8
    cycle_duration = 1.0
    
    while True:
        start_time = time.time()
        active_start = time.time()
        

        while (time.time() - start_time) < cycle_duration:

            while (time.time() - active_start) < (cycle_duration * duty_cycle):
                with torch.no_grad():
                    _ = a @ b
                torch.cuda.synchronize(device)
            
            time.sleep(cycle_duration * (1 - duty_cycle))
            active_start = time.time()

if __name__ == '__main__':
    processes = []

    if len(sys.argv) < 2:
        print("Please provide a list of numbers.")
        sys.exit(1)

    numbers = sys.argv[1]
    gpu_ids = [int(num) for num in numbers.split(',')]
    
    print('GPU ids: ', gpu_ids)
    
    for gpu_id in gpu_ids:
        p = mp.Process(target=gpu_worker, args=(gpu_id,))
        p.start()
        processes.append(p)
    
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        for p in processes:
            p.terminate()