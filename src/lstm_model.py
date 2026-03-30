import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import os

# -------------------------
# 1. Read dataset
# -------------------------
with open("data/dataset.txt", "r") as f:
    lines = f.read().splitlines()

pairs = [line.split(" -> ") for line in lines]

# -------------------------
# 2. Tokenizers
# -------------------------
all_chars = sorted(list(set("".join([inp + out for inp, out in pairs]))))
char2idx = {c:i+1 for i,c in enumerate(all_chars)}  # 0 for padding
idx2char = {i:c for c,i in char2idx.items()}
vocab_size = len(char2idx)+1

max_input_len = max(len(inp) for inp, out in pairs)
max_output_len = max(len(out) for inp, out in pairs)

def encode(s, max_len):
    arr = [char2idx[c] for c in s]
    arr += [0]*(max_len - len(arr))
    return arr

def decode(arr):
    return "".join([idx2char[i] for i in arr if i != 0])

# -------------------------
# 3. Dataset
# -------------------------
class TaylorDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        inp, out = self.pairs[idx]
        x = torch.tensor(encode(inp, max_input_len), dtype=torch.long)
        y = torch.tensor(encode(out, max_output_len), dtype=torch.long)
        return x, y

dataset = TaylorDataset(pairs)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# -------------------------
# 4. LSTM with Attention
# -------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
    
    def forward(self, x):
        x = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size*2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs):
        # hidden: [1, batch, hidden], encoder_outputs: [batch, seq_len, hidden]
        hidden = hidden.permute(1, 0, 2)  # [batch, 1, hidden]
        seq_len = encoder_outputs.size(1)
        hidden = hidden.repeat(1, seq_len, 1)  # [batch, seq_len, hidden]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = torch.softmax(self.v(energy), dim=1)
        return attention

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden, cell, encoder_outputs):
        x = self.embedding(x)  # [batch, 1, embed]
        attn_weights = self.attention(hidden, encoder_outputs)  # [batch, seq_len, 1]
        context = torch.bmm(attn_weights.permute(0,2,1), encoder_outputs)  # [batch,1,hidden]
        x = torch.cat((x, context), dim=2)
        outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
        predictions = self.fc(outputs)
        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, max_output_len):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.max_output_len = max_output_len
    
    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        device = src.device
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # start token: padding (0) for simplicity
        inputs = torch.zeros(batch_size,1,dtype=torch.long).to(device)
        outputs_seq = torch.zeros(batch_size, self.max_output_len, vocab_size).to(device)
        
        for t in range(self.max_output_len):
            outputs, hidden, cell = self.decoder(inputs, hidden, cell, encoder_outputs)
            outputs_seq[:,t,:] = outputs.squeeze(1)
            top1 = outputs.argmax(2)
            if trg is not None and random.random() < teacher_forcing_ratio:
                inputs = trg[:,t].unsqueeze(1)
            else:
                inputs = top1
        return outputs_seq

# -------------------------
# 5. Instantiate Model
# -------------------------
embed_size = 64
hidden_size = 128

encoder = Encoder(vocab_size, embed_size, hidden_size)
decoder = Decoder(vocab_size, embed_size, hidden_size)
model = Seq2Seq(encoder, decoder, max_output_len)

# -------------------------
# 6. Training
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):  # small epochs for testing
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x, y)
        output = output.view(-1, vocab_size)
        y = y.view(-1)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

# -------------------------
# 7. Test / Sample Prediction
# -------------------------
model.eval()
sample_inp, _ = random.choice(pairs)
x_test = torch.tensor([encode(sample_inp, max_input_len)], dtype=torch.long).to(device)

with torch.no_grad():
    pred = model(x_test, trg=None, teacher_forcing_ratio=0.0)
    pred_chars = pred.argmax(-1).cpu().numpy()[0]
    print("Input:", sample_inp)
    print("Predicted Taylor:", decode(pred_chars))