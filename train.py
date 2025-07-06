import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

###################################################
#                (hyper) Parameters               #
###################################################
torch.manual_seed(1337)
DATASET_PATH = "C:/Users/doyez/Downloads/wikisent2.txt"
PERCENTAGE_TRAINING = 0.9
CONTEXT_LENGTH = 256
BATCHSIZE = 4
LEARNINGRATE = 1e-4
EPOCHS = 10000
VAL_INTERVAL = 1000
N_EMBD = 32
NB_LAYERS = 5
NB_HEAD = 4
DROPOUT = 0.3
####################################################


# Set the GPU if available:
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[i] device used: {device}")

# Read the dataset:
with open(DATASET_PATH, "r") as f:
  text = f.read()

# Get all the caracters available:
chars = sorted(list(set(text)))
vocab_size = len(chars)

# 2 dictionnaries:
#stoi maps caracters to integers: "c" : 1 for example
# itos maps integers to caracters 1: "c"

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

# We can use tiktoken lib also, but keep it simple:
def encode(s):
  return [stoi[c] for c in s]

def decode(l):
  return "".join([itos[i] for i in l])

########################################
#         Dataset preparation          #
########################################

print("\033[33m[i] Dataset Preparation started:\033[0m")
# Encoding our dataset:
data = torch.tensor(encode(text), dtype=torch.long)

#Split the dataset:
n = int(PERCENTAGE_TRAINING * len(data))
train_data = data[:n]
val_data = data[n:]

# Input:
x = train_data[:CONTEXT_LENGTH]
#Expected output:
y = train_data[1:CONTEXT_LENGTH+1]

# Create the batchs:
def get_batch(split):
    data = train_data if split == "train" else val_data
    # Starting position:
    ix = torch.randint(len(data) - CONTEXT_LENGTH, (BATCHSIZE,))

    x = torch.stack([data[i:i+CONTEXT_LENGTH] for i in ix])
    y = torch.stack([data[i+1:i+CONTEXT_LENGTH+1] for i in ix])
    return x, y

xb, yb = get_batch("train")
print("\033[32m[i] Data ready.\033[0m")

###########################################
#                Model                    #
###########################################
# Self attention mecanism into the head:

class Head(nn.Module):
    def __init__(self, head_size):

        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias = False)
        self.query = nn.Linear(N_EMBD, head_size, bias = False)
        self.value = nn.Linear(N_EMBD, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # This is self attention:
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        weights = q @ k.transpose(-2,-1) * (k.shape[-1]**-0.5) # For nomalisation
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim= -1)
        weights = self.dropout(weights)

        out = weights @ v
        return out

# To have a better LLM, use multiples heads:
class MultiHead(nn.Module):
    def __init__(self, nb_head, head_size):
        super().__init__()  
        # Just create multiples heads:
        self.heads = nn.ModuleList([Head(head_size) for _ in range(nb_head)])
        self.proj = nn.Linear(head_size * nb_head, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim= -1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.ff = nn.Sequential(
          nn.Linear(size, 4 * size),
          nn.ReLU(), 
          nn.Linear(4 * size, size),
          nn.Dropout(DROPOUT)
        )

    def forward(self, x):
      return self.ff(x)
    
class Blocks(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.multihead_att = MultiHead(n_head, head_size)
        self.ff = FeedForward(n_embed)
        self.layerNorm1 = nn.LayerNorm(n_embed)
        self.layerNorm2 = nn.LayerNorm(n_embed)

    
    def forward(self, x):
        # Change from Attention is all you need:
        # LayerNorm are befoore multihead and feed forward!
        # x + ... for skip connection
        x = x + self.multihead_att(self.layerNorm1(x))
        x = x + self.ff(self.layerNorm2(x))
        return x
    

class NanoGPT(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
    self.pos_embedding_table = nn.Embedding(CONTEXT_LENGTH, N_EMBD)
    
    self.blocks = nn.Sequential(*[Blocks(n_embed= N_EMBD, n_head= NB_HEAD) for _ in range (NB_LAYERS)])
    self.layer_norm = nn.LayerNorm(N_EMBD)
    
    self.lm_head = nn.Linear(N_EMBD, vocab_size)

  def forward(self, idx, targets=None):
    B, T= idx.shape
    token_emb = self.token_embedding_table(idx) # shape is B, T, C, where C is the embedding
    pos_embd = self.pos_embedding_table(torch.arange(T, device= device))
    x = token_emb + pos_embd 
    x = self.blocks(x)
    x = self.layer_norm(x)

    logits = self.lm_head(x)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      # Doc Pytorch: cross_entropy needs data (minibatch,C)  = (B*T, C)
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)

      loss = F.cross_entropy(logits, targets)
    return logits, loss
  
  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      # Crop because we only work with the last tokens from context_length:
      idx_crop = idx[:, -CONTEXT_LENGTH:]
      logits, loss = self(idx_crop)
      logits = logits[:, -1, :]

      proba = F.softmax(logits, dim=-1)
      # Add creativity: choose with the probability with multinomial:
      # If we have a tensor [0.9, 0.05, 0.7], we have great chance that multinomial choose 1 or 3, due to high probability
      idx_next = torch.multinomial(proba, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx

@torch.no_grad()
def estimate_loss():
    out = {}
    # Put in eval, to desactivate the drop out, batchnorm...
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(VAL_INTERVAL)
        for k in range(VAL_INTERVAL):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
###############################################
##                  TRAINING                 ##
###############################################

model = NanoGPT()
m = model.to(device) # load the model in GPU if possible

optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNINGRATE)
print("\033[<33>m<[i] Training started:>\033[0m")
for iter in range(EPOCHS):
    if iter % VAL_INTERVAL or iter == EPOCHS - 1:
        loss = estimate_loss()
        print(f"Epoch {iter}/{EPOCHS}:\n train loss: {loss["train"]:.4f}, val loss: {loss["val"]:.4f}")
    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none= True)

    loss.backward()
    optimizer.step()


###################################################
##                   TEST                        ##
###################################################

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))