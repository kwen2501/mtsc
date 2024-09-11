import os
import torch
import torch.distributed as dist

# One pair of server and client always have group size of 2
group_size = 2

def run(gid):
    print(f"Client {gid}: start init")
    device = torch.device(f'cuda:{gid}')
    filename = f"file_{gid}"
    store = dist.FileStore(filename, group_size)
    dist.init_process_group(
        backend='nccl',
        rank=1,  # client rank is always 1
        world_size=group_size,
        store=store,
        device_id=device,
    )
    x = torch.ones(1, device=device) * (-1)
    dist.recv(x, src=0)
    print(f"Client {gid}: got {x}")
    dist.destroy_process_group()
    print(f"Client {gid}: clean exit")


def main():
    # The example uses torchrun to launch multiple clients, e.g.
    # torchrun --nproc-per-node=2 client.py
    # Abusing the "RANK" env var to specify the client id.
    # RANK=0, 1, 2, ... correspond to server-client pairs 0, 1, 2, ...

    # If you are using python to launch the clients, please specify `RANK` like:
    # RANK=0 python client.py
    # RANK=1 python client.py
    # RANK=2 python client.py
    # ...
    gid = os.environ["RANK"]
    run(gid)


if __name__ == "__main__":
    main()
