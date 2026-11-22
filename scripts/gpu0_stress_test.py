import torch, time, subprocess, sys

GPU_ID = 0
DURATION_SECONDS = 600
INTERVAL = 5

torch.cuda.set_device(GPU_ID)
MB = 1024 * 1024
total_bytes = torch.cuda.get_device_properties(GPU_ID).total_memory
reserve = int(total_bytes * 0.55)
alloc_size = (total_bytes - reserve) // 4
cols = 4096
rows = alloc_size // cols
x = torch.randn(rows, cols, dtype=torch.float32, device='cuda')
N = 16384
_ = torch.mm(x[:N, :], x[:N, :].T)
torch.cuda.synchronize()

start = time.time()
next_sample = start
temps = []
utils = []
mems = []

while True:
    now = time.time()
    if now - start >= DURATION_SECONDS:
        break

    t0 = time.time()
    for _ in range(20):
        y = torch.mm(x[:N, :], x[:N, :].T)
        z = torch.relu(y)
        torch.cuda.synchronize()
    t1 = time.time()

    if now >= next_sample:
        try:
            out = subprocess.check_output([
                'nvidia-smi',
                '--query-gpu=index,temperature.gpu,utilization.gpu,memory.used,memory.total',
                '--format=csv,noheader,nounits',
                '-i', str(GPU_ID),
            ], text=True)
            parts = [p.strip() for p in out.split(',')]
            temps.append(int(parts[1]))
            utils.append(int(parts[2]))
            mems.append(int(parts[3]))
            elapsed = now - start
            print(f'[t={elapsed:6.1f}s] temp={parts[1]:>3} C  util={parts[2]:>3}%  mem={parts[3]:>6}/{parts[4]} MB  iter={(t1-t0):.3f}s')
        except Exception as e:
            print('nvidia-smi error:', e, file=sys.stderr)
        next_sample += INTERVAL

    time.sleep(max(0, next_sample - time.time()))

print('\n--- Summary ---')
print(f'Duration: {DURATION_SECONDS}s')
print(f'Temp min/avg/max: {min(temps)} / {sum(temps)/len(temps):.1f} / {max(temps)} C')
print(f'Util min/avg/max: {min(utils)} / {sum(utils)/len(utils):.1f} / {max(utils)} %')
print(f'Mem used: {mems[-1]} MB')
print(f'GPU 0 final memory: {subprocess.check_output(["nvidia-smi","--query-gpu=memory.used,memory.total","--format=csv,noheader,nounits","-i","0"], text=True).strip()} MB')
