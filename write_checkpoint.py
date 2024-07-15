import numpy as np
import torch

DIM = 4096
VOCAB_SIZE = 65536

# embedding = np.zeros((VOCAB_SIZE, DIM), np.float32)
# norm = np.zeros(VOCAB_SIZE, np.float32)
# output = np.zeros((DIM, VOCAB_SIZE), np.float32)
# ckpt = np.concatenate([embedding.flatten(), norm.flatten(), output.flatten()])
# 
# ckpt = np.arange(VOCAB_SIZE * DIM * 2 + VOCAB_SIZE, dtype=np.float32)

ckpt = torch.load("chameleon_7b.pth", map_location="cpu") # TODO
ckpt = {k: v.float() for k, v in ckpt.items()}

ckpt = [ckpt['tok_embeddings.weight'], ckpt["norm.weight"], ckpt["output.weight"].T]

ckpt = np.concatenate([x.flatten().numpy() for x in ckpt])

print(len(ckpt))

with open('ckpt.bin', 'wb') as f:
    f.write(ckpt.tobytes())