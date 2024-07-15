import ctypes
import numpy as np
import torch

DIM = 4096
VOCAB_SIZE = 65536

B = 2
T = 256
C = DIM

return_dbg = True

lib = ctypes.CDLL('./model.so')

lib.model.argtypes = [ctypes.POINTER(ctypes.c_uint16)]
if return_dbg:
    lib.model.restype = ctypes.POINTER(ctypes.c_float)
else:
    lib.model.restype = ctypes.c_float

lib.init()

ckpt = torch.load('chameleon_7b.pth', map_location='cpu')
embeddings = ckpt['tok_embeddings.weight'].float().numpy()

while True:
    data = list(range(B*T))
    data = np.array(data, np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))

    if return_dbg:
        dbg = lib.model(data)
        dbg = ctypes.cast(dbg, ctypes.POINTER(ctypes.c_float * B * T * DIM))
        dbg = np.frombuffer(dbg.contents, dtype=np.float32)
        dbg = dbg.reshape(B, T, DIM)
        print(dbg)
        import ipdb; ipdb.set_trace()
    else:
        loss = lib.model(data)
        print(f"loss {loss}")


    break