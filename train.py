import torch

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

