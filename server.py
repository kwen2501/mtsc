import threading
import torch
import torch.distributed as dist
from torch.testing._internal.distributed.multi_threaded_pg import _install_threaded_pg

# Assuming one server + one client form a group.
# If one server serves multiple clients, bump this number.
group_size = 2

# Routine of each server thread
# 1. init process group
# 2. send its thread id to client
# 3. destroy process group
# Args:
#     gid: server-client pair id
def run(gid):
    print(f"Server {gid}: start init")
    # This line is a must for each thread to have its own c10d states
    torch._C._distributed_c10d._set_thread_isolation_mode(True)
    _install_threaded_pg()
    device = torch.device(f'cuda:{gid}')
    filename = f"file_{gid}"
    store = dist.FileStore(filename, group_size)
    dist.init_process_group(
        backend='nccl',
        rank=0,  # server rank is always 0
        world_size=group_size,
        store=store,
        device_id=device,
    )
    x = torch.ones(1, device=device) * gid
    print(f"Server {gid}: start collective")
    dist.send(x, dst=1)
    dist.destroy_process_group()
    print(f"Server {gid}: clean exit")


# To launch the server side, just do:
#     python server.py
# It will start 4 server threads, each thread serving a client.
def main():
    num_servers = 4
    for gid in range(num_servers):
        t = threading.Thread(target=run, args=(gid,))
        t.start()


if __name__ == "__main__":
    main()
