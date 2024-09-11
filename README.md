# mtsc
A simple multi-thread server-client model using PyTorch c10d

![Screenshot 2024-09-11 at 4 31 36 PM](https://github.com/user-attachments/assets/8d41079e-c503-46fb-8d31-3a53b1527c14)

# Server Side
To launch the server side, just do:
```
rm file_*
python server.py
```
It will start 4 server threads, each thread serving a client.

# Client Side
For the client side, open a new terminal and offset the `CUDA_VISIBLE_DEVICES` because devices 0-3 are used as servers. 
```
export CUDA_VISIBLE_DEVICES=4,5,6,7
```
(You do not need this step if the clients are on a different host than the servers.)

To launch multiple clients, you can do either (i) or (ii) below:

(i) Using torchrun:
```
torchrun --nproc-per-node 4 client.py
```
(ii) Using python:
```
RANK=0 python client.py &
RANK=1 python client.py &
RANK=2 python client.py &
RANK=3 python client.py &
```
(Case (ii) mimics the case in which clients join dynamically.)

# Demo
Server:
```
$ rm file_*; python server.py
Server 0: start init
Server 1: start init
Server 2: start init
Server 3: start init
NCCL version 2.21.5+cuda12.0
Server 0: start collective
Server 2: start collective
Server 3: start collective
Server 1: start collective
Server 0: clean exit
Server 3: clean exit
Server 2: clean exit
Server 1: clean exit
```

Clients:
```
$ CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc-per-node 4 client.py
Client 3: start init
Client 0: start init
Client 2: start init
Client 1: start init
Client 0: got tensor([0.], device='cuda:0')
Client 0: clean exit
Client 3: got tensor([3.], device='cuda:3')
Client 2: got tensor([2.], device='cuda:2')
Client 3: clean exit
Client 2: clean exit
Client 1: got tensor([1.], device='cuda:1')
Client 1: clean exit
```
