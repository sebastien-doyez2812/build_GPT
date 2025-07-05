import torch
import torch.nn as nn
from torch.nn import functional as F

###################################################
#                (hyper) Parameters               #
###################################################
torch.manual_seed(1337)
DATASET_PATH = "/content/drive/MyDrive/Colab Notebooks/build_a_GPT/wikisent2.txt"
PERCENTAGE_TRAINING = 0.9
CONTEXT_LENGTH = 8
BATCHSIZE = 4
LEARNINGRATE = 1e-4
EPOCHS = 10000
N_EMBD = 32
NB_CHANNELS = 4
####################################################


# Set the GPU if available:
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Read the dataset:
with open(DATASET_PATH, "r") as f:
  text = f.read()

# Test:
print(text[:500])

# Get all the caracters available:
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(chars)
print(vocab_size)

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

# Test:
print(encode("Hey, i'm Seb"))
print(decode(encode("Hey, i'm Seb")))

########################################
#         Dataset preparation          #
########################################

# Encoding our dataset:
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)

#Split the dataset:
n = int(PERCENTAGE_TRAINING * len(data))
train_data = data[:n]
val_data = data[n:]

# Input:
x = train_data[:CONTEXT_LENGTH]
#Expected output:
y = train_data[1:CONTEXT_LENGTH+1]

#  Test:
for t in range(CONTEXT_LENGTH):
  context = x[:t+1]
  target = y[t]
  print(f"when input is {context} the target: {target}")

# Create the batchs:

def get_batch(split):
  data = train_data if split == "train" else val_data
  # Starting position:
  ix = torch.randint(len(data) - CONTEXT_LENGTH, (BATCHSIZE,))

  x = torch.stack([data[i:i+CONTEXT_LENGTH] for i in ix])
  y = torch.stack([data[i+1:i+CONTEXT_LENGTH+1] for i in ix])
  return x, y

xb, yb = get_batch("train")


###########################################
#                Model                    #
###########################################
# Self attention mecanism into the head:

class Head(nn.Module):
    def __init__(self, head_size):

        super().__init__()
        self.key = key = nn.Linear(N_EMBD, head_size, bias = False)
        self.query = nn.Linear(N_EMBD, head_size, bias = False)
        self.value = nn.Linear(N_EMBD, head_size, bias = False)
        self.register_buffer('tril', torch.trill(torch.ones[CONTEXT_LENGTH, CONTEXT_LENGTH]))

    def forward(self, x):
      # This is self attention:
      B, T, C = x.shape
      k = self.key(x)
      q = self.query(x)
      v = self.value(x)

      weights = q @ k.transpose(-2,1) * (C**-0.5) # For nomalisation
      weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
      weights = F.softmax(weights)

      out = weights @ v
      return out

# To have a better LLM, use multiples heads:
class MultiHead(nn.Module):
  def __init__(self, nb_head, head_size):
    super().__init__()  
    # Just create multiples heads:
    self.heads = nn.ModuleList(Head(head_size) for _ in range(nb_head))

    def forward(self, x):
      return torch.cat([h(x) for h in self.heads], dim= 1)

class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
    self.pos_embedding_table = nn.Embedding(CONTEXT_LENGTH, N_EMBD)
    
    self.self_att_head = MultiHead(NB_CHANNELS, N_EMBD)
    self.lm_head = nn.linear(N_EMBD, vocab_size)

  def forward(self, idx, targets=None):
    B, T, C = idx.shape()
    token_emb = self.token_embedding_table(idx) # shape is B, T, C, where C is the embedding
    pos_embd = self.pos_embedding_table(torch.arange(T, device= device))
    x = token_emb + pos_embd 
    x = self.self_att_head(x)
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



m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)

