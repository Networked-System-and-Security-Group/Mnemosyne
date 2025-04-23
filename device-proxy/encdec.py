import torch
import torch.nn as nn
import torch.optim as optim
import time
import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, Q, K, V):
        batch, seq_len, _ = Q.shape
        Q = Q.view(batch, seq_len, self.heads, self.head_dim)
        K = K.view(batch, seq_len, self.heads, self.head_dim)
        V = V.view(batch, seq_len, self.heads, self.head_dim)
        
        energy = torch.einsum("bqhd,bkhd->bhqk", Q, K) / (self.head_dim ** 0.5)
        attention = torch.softmax(energy, dim=-1)
        
        out = torch.einsum("bhqk,bkhd->bqhd", attention, V)
        out = out.reshape(batch, seq_len, self.embed_size)
        return self.fc_out(out)

class EncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_size)
        )
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        attn_output = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, embed_size, heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.encoder_attention = SelfAttention(embed_size, heads)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_size)
        )
        self.norm3 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output):
        self_attn_output = self.self_attention(x, x, x)
        x = self.norm1(x + self_attn_output)
        
        enc_attn_output = self.encoder_attention(x, encoder_output, encoder_output)
        x = self.norm2(x + enc_attn_output)
        
        ff_output = self.ff(x)
        x = self.norm3(x + ff_output)
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, heads, num_layers, ff_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(1000, embed_size)
        self.layers = nn.ModuleList(
            [EncoderLayer(embed_size, heads, ff_dim) for _ in range(num_layers)]
        )
    
    def forward(self, x):
        positions = torch.arange(x.shape[1], device=device).unsqueeze(0)
        x = self.embed(x) + self.pos_embed(positions)
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, heads, num_layers, ff_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(1000, embed_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(embed_size, heads, ff_dim) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(embed_size, vocab_size)
    
    def forward(self, x, encoder_output):
        positions = torch.arange(x.shape[1], device=device).unsqueeze(0)
        x = self.embed(x) + self.pos_embed(positions)
        for layer in self.layers:
            x = layer(x, encoder_output)
        return self.fc(x)

class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_size, heads, num_layers, ff_dim):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_size, heads, num_layers, ff_dim)
        self.decoder = Decoder(vocab_size, embed_size, heads, num_layers, ff_dim)
    
    def forward(self, src, trg):
        encoder_output = self.encoder(src)
        output = self.decoder(trg, encoder_output)
        return output

vocab_size = 50
seq_len = 10
embed_size = 64
heads = 4
num_layers = 16  
ff_dim = 256    
batch_size = 32
epochs = 100

model = Transformer(vocab_size, embed_size, heads, num_layers, ff_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

start_time = time.time()

for epoch in range(epochs):
    src = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    trg = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    
    outputs = model(src, trg)
    loss = criterion(outputs.view(-1, vocab_size), trg.view(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

end_time = time.time()
print(f"Time: {end_time - start_time:.2f}")