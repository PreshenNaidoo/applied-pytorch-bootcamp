import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt
import os
import sys
import re

device = "cuda" if torch.cuda.is_available() else "cpu"

class gpt_config:

    batch_size: int = 32
    context_length: int = 128
    embed_dim: int = 384
    mlp_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    learning_rate: float = 1e-4
    num_samples: int = 200
    epochs: int = 1
    random_state: int = 42
    padding_idx = 0

    file_path: str = "data/input.txt"

    vocab_size: int | None = None
    str_to_int: dict | None = None
    int_to_str: dict | None = None


def basic_word_tokenize(text: str) -> list[str]:
    """Simple fallback tokenizer for word-level experiments."""
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def read_raw_text_file(config: gpt_config) -> str:
    with open(file=config.file_path, mode="r", encoding="utf-8") as f:
        raw_text = f.read()
    return raw_text

def split_text_corpus(raw_text: str) -> dict[str, str]:
    
    num_train, num_val = int(len(raw_text) * 0.6), int(len(raw_text) * 0.2)

    train_text = raw_text[:num_train]
    val_text = raw_text[num_train: num_train+num_val]
    test_text = raw_text[num_train+num_val:]

    return {"train_text": train_text, "val_text": val_text, "test_text": test_text}

class WordTokenDataset(torch.utils.data.Dataset):
    def __init__(self, tokens: list[str], context_length:int, vocab_dict: dict[str, int]):

        self.tokens = tokens
        self.num_tokens = len(tokens)
        self.context_length = context_length
        self.vocab_dict = vocab_dict
        self.unk_id = 1        

    def __len__(self) -> int:
        return self.num_tokens - self.context_length

    def __getitem__(self, index: int) -> dict[str, torch.tensor]:
        
        tokens = self.tokens[index : index + self.context_length]
        targets = self.tokens[index + 1: index + 1 + self.context_length]

        token_ids, target_ids = [], []
        for token in tokens:
            if token in self.vocab_dict:
                token_ids.append(self.vocab_dict[token])
            else:
                token_ids.append(self.unk_id)

        for target_token in targets:
            if target_token in self.vocab_dict:
                target_ids.append(self.vocab_dict[target_token])
            else:
                target_ids.append(self.unk_id)

        return {"input_ids": torch.tensor(token_ids, dtype=torch.long), "target_ids": torch.tensor(target_ids, dtype=torch.long)}


def load_dataloaders(config: gpt_config) -> dict[str, torch.utils.data.DataLoader]:

    raw_text = read_raw_text_file(config)
    data_dict = split_text_corpus(raw_text)

    all_words = basic_word_tokenize(raw_text)
    vocabulary = sorted(set(all_words))
    vocabulary.insert(0, "[PAD]")
    vocabulary.insert(1, "[UNK]")
    vocab_size = len(vocabulary)

    str_to_int = {w:i for i, w in enumerate(vocabulary)}
    int_to_str = {i:w for w, i in str_to_int.items()}

    config.vocab_size = vocab_size
    config.str_to_int = str_to_int
    config.int_to_str = int_to_str

    train_ds = WordTokenDataset(tokens=basic_word_tokenize(data_dict["train_text"]), context_length=config.context_length, vocab_dict=str_to_int)
    val_ds = WordTokenDataset(tokens=basic_word_tokenize(data_dict["val_text"]), context_length=config.context_length, vocab_dict = str_to_int)
    test_ds = WordTokenDataset(tokens=basic_word_tokenize(data_dict["test_text"]), context_length=config.context_length, vocab_dict= str_to_int)

    return {"train": torch.utils.data.DataLoader(dataset=train_ds, batch_size=config.batch_size, shuffle=True),
            "val": torch.utils.data.DataLoader(dataset=val_ds, batch_size=config.batch_size),
            "test": torch.utils.data.DataLoader(dataset=test_ds, batch_size=config.batch_size)}


def build_causal_mask(context_length: int) -> torch.Tensor:
    
    ones = torch.ones(size=(context_length, context_length), dtype=torch.bool)
    mask = torch.tril(ones)
    mask = mask.unsqueeze(dim=0).unsqueeze(dim=1)                       # shape = (1, 1, context_length, context_length)

    return mask

def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    
    batch_size, num_heads, context_length, head_dim = query.shape

    scores = torch.matmul(query, key.transpose(dim0=2, dim1=3))         # (32, 4, 128, 96) @ (32, 4, 96, 128) = (32, 4, 128, 128)
    scores = scores/sqrt(head_dim)

    weights = scores.masked_fill(~mask, value=float("-inf"))            # (32, 4, 128, 128)
    weights = weights.softmax(dim=-1)                                   # (32, 4, 128, 128)

    output = weights @ value                                            # (32, 4, 128, 128) @ (32, 4, 128, 96) = (32, 4, 128, 96)

    return output


class MaskedSelfAttention(nn.Module):
    def __init__(self, config: gpt_config):
        super().__init__()

        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads

        if self.embed_dim % self.num_heads != 0:
            raise ValueError("Embedded dimension is not exactly divsible by number of heads.")

        self.head_dim = self.embed_dim // self.num_heads

        self.query = nn.Linear(in_features = config.embed_dim, out_features=config.embed_dim)
        self.key = nn.Linear(in_features=config.embed_dim, out_features=config.embed_dim)
        self.value = nn.Linear(in_features=config.embed_dim, out_features=config.embed_dim)

        self.proj = nn.Linear(in_features=config.embed_dim, out_features=config.embed_dim)

    def forward(self, x: torch.tensor, mask:torch.tensor) -> torch.tensor:
        
        batch_size, context_length, embed_dim = x.shape                                     # (32, 128, 384)

        query = self.query(x)                                                               # (32, 128, 384)
        key = self.key(x)                                                                 # (32, 128, 384)
        value = self.value(x)                                                               # (32, 128, 384)

        query = query.reshape(batch_size, context_length, self.num_heads, self.head_dim)    # (32, 128, 4, 96)
        key = key.reshape(batch_size, context_length, self.num_heads, self.head_dim)        # (32, 128, 4, 96)
        value = value.reshape(batch_size, context_length, self.num_heads, self.head_dim)    # (32, 128, 4, 96)

        query = query.permute(dims=(0, 2, 1, 3))                                            # (32, 4, 128, 96)
        key = key.permute(dims=(0, 2, 1, 3))                                                # (32, 4, 128, 96)
        value = value.permute(dims=(0, 2, 1, 3))                                            # (32, 4, 128, 96)

        att_output = scaled_dot_product_attention(query, key, value, mask)                  # (32, 4, 128, 96)

        att_output = att_output.permute(dims=(0, 2, 1, 3))                                  # (32, 128, 4, 96)
        att_output = att_output.reshape(shape=(batch_size, context_length, self.num_heads*self.head_dim))   # (32, 128, 384)

        proj = self.proj(att_output)                                                        # (32, 128, 384)

        return proj


class GPTBlock(nn.Module):
    def __init__(self, config: gpt_config):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(normalized_shape=config.embed_dim)
        self.masked_self_attention = MaskedSelfAttention(config)

        self.layer_norm2 = nn.LayerNorm(normalized_shape=config.embed_dim)
        self.mlp = nn.Sequential(nn.Linear(in_features=config.embed_dim, out_features=config.mlp_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(in_features=config.mlp_dim, out_features=config.embed_dim))
        


    def forward(self, x: torch.tensor, mask: torch.tensor) -> torch.tensor:        

        residual = x
        x = self.layer_norm1(x)
        x = self.masked_self_attention(x, mask)
        x = x + residual

        residual = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = x + residual

        return x

class MiniGPT(nn.Module):
    def __init__(self, config: gpt_config):
        super().__init__()

        self.causal_mask = build_causal_mask(config.context_length)        

        self.token_embeddings = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embed_dim, padding_idx=config.padding_idx)

        self.pos_embeddings = nn.Embedding(num_embeddings=config.context_length, embedding_dim=config.embed_dim)

        self.decoder_blocks = nn.ModuleList([GPTBlock(config) for _ in range(config.num_layers)])

        self.layer_norm = nn.LayerNorm(normalized_shape=config.embed_dim)

        self.classification_head = nn.Linear(in_features=config.embed_dim, out_features=config.vocab_size)

    def forward(self, token_ids: torch.tensor) -> torch.tensor:
        
        # token_ids shape is (batch_size, context_length)       # sequence_length when inferring, context_length when training

        token_embeddings = self.token_embeddings(token_ids)                                 # shape is (32, sequence_length, 384)

        batch_size, sequence_length, embed_dim = token_embeddings.shape

        positions = torch.arange(start=0, end=sequence_length, step=1, device = token_ids.device)
        positions = positions.unsqueeze(dim=0)                                              # shape is (1, sequence_length)
        positions = positions.expand(size = (batch_size, sequence_length))                  # shape is (32, sequence_length)

        positional_embeddings = self.pos_embeddings(positions)                              # shape is (32, sequence_length, 384)

        embeddings = token_embeddings + positional_embeddings                               # shape is (32, sequence_length, 384)
        
        self.causal_mask = self.causal_mask.to(token_ids.device)
        mask = self.causal_mask[:, :, :sequence_length, :sequence_length]                   # shape is (1, 1, seq_length, seq_length)

        features = embeddings
        for decoder in self.decoder_blocks:
            features = decoder(features, mask)

        features = self.layer_norm(features)                                                # (32, sequence_length, 384)

        logits = self.classification_head(features)                                         # (32, sequence_length, vocab_size)

        return logits


def build_training_objects(config: gpt_config) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    model = MiniGPT(config)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr=config.learning_rate, params=model.parameters())

    return model, loss_fn, optimizer

def train_one_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, float]:
    model.train()

    losses = []

    for data_dict in loader:
        input_ids = data_dict["input_ids"]
        target_ids = data_dict["target_ids"]

        input_ids = input_ids.to(device)            # (32, 128)
        target_ids = target_ids.to(device)          # (32, 128)

        optimizer.zero_grad()

        logits = model(input_ids)                   # (32, 128, 13333)

        batch_size, context_length, vocab_size = logits.shape

        loss = loss_fn(logits.reshape(batch_size * context_length, vocab_size), target_ids.reshape(batch_size * context_length))
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    mean_loss = torch.tensor(losses).mean()
    perp = torch.exp(mean_loss)

    return {"loss": mean_loss.item(), "perplexity": perp.item()}

@torch.no_grad()
def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, loss_fn: nn.Module) -> dict[str, float]:
    
    model.eval()

    losses = []

    for data_dict in loader:
        input_ids = data_dict["input_ids"]
        target_ids = data_dict["target_ids"]

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(input_ids)
        batch_size, context_length, vocab_size = logits.shape

        loss = loss_fn(logits.reshape(batch_size * context_length, vocab_size), target_ids.reshape(batch_size * context_length))
        losses.append(loss.item())

    mean_loss = torch.tensor(losses).mean()
    perp = torch.exp(mean_loss)

    return {"loss": mean_loss.item(), "perplexity": perp.item()}

def generate_text(model: nn.Module, start_text: str, config: gpt_config) -> str:

    temperature = 0.5
    use_temperature = True
    
    if not start_text:
        start_text = "Hello world"

    start_tokens = basic_word_tokenize(start_text)
    start_tokens = start_tokens[-config.context_length:]

    encode = lambda st: [config.str_to_int.get(word, 1) for word in st]
    decode = lambda l: " ".join([config.int_to_str[i] for i in l])

    model_device = next(model.parameters()).device

    token_ids = torch.tensor(encode(start_tokens), dtype=torch.long, device = model_device)
    token_ids = token_ids.unsqueeze(dim=0) # add a fake batch dimension, shape is now (1, sequence length)

    generated = []
    for i in range(config.num_samples):

        logits = model(token_ids)               # shape is batch_size, sequence_length, vocab_size
        last_token_vec = logits[:, -1, :]       # shape is (1, vocab_size)

        if use_temperature:
            prob = torch.softmax(last_token_vec/temperature, dim=-1)
            last_token_id = torch.multinomial(input = prob, num_samples=1)      # (1, 1)
        else:
            last_token_id = last_token_vec.argmax(dim=-1).long()                #scalar
            last_token_id = last_token_id.unsqueeze(dim=0).unsqueeze(dim=0)     # shape is now (1, 1)

        generated.append(last_token_id.squeeze().item())        

        token_ids = torch.concat([token_ids, last_token_id], dim=1)
        token_ids = token_ids[:, -config.context_length:]

    generated_text = start_text + " " + decode(generated)

    return generated_text

def run_training(config: gpt_config) -> dict[str, float]:
    loaders = load_dataloaders(config)
    model, loss_fn, optimizer = build_training_objects(config)

    for i in range(config.epochs):

        train_dict = train_one_epoch(model, loaders["train"], loss_fn, optimizer)
        val_dict = evaluate(model, loaders["val"], loss_fn)

        print(f"Epoch {i+1} -> loss: {train_dict['loss']}, perp: {train_dict['perplexity']}, val_loss: {val_dict['loss']}, val_perp: {val_dict['perplexity']}")

    test_dict = evaluate(model, loaders["test"], loss_fn)
    print(f"Test results -> loss: {test_dict['loss']}, perp: {test_dict['perplexity']}")

    print(f"\n GENERATED: \n {generate_text(model=model, start_text = 'Hello world', config=config)}")

    return {"train_loss": train_dict["loss"], "train_perp": train_dict["perplexity"], "val_loss": val_dict["loss"], "val_perp": val_dict["perplexity"], "test_loss": test_dict["loss"], "test_perp": test_dict["perplexity"]}

def main():
    config = gpt_config()
    train_results = run_training(config)
    print(f"Results: {train_results}")

if __name__ == "__main__":
    main()