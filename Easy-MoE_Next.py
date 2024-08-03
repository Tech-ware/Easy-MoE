import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge 

# ---------- Device Configuration ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Utility Functions ----------
def positional_encoding(seq_len, d_model, device):
    pos = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(device)
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe.unsqueeze(0)

# ---------- Model Definitions ----------
class TransformerExpert(nn.Module):
    def __init__(self, input_size, d_model, output_size, nhead, dim_feedforward, num_encoder_layers=1, dropout=0.1):
        super(TransformerExpert, self).__init__()
        self.d_model = d_model
        self.input_fc = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_fc = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_fc(x) * math.sqrt(self.d_model) + positional_encoding(x.size(1), self.d_model, x.device)
        x = self.dropout(x) # Apply dropout after positional encoding
        transformer_output = self.transformer_encoder(x)
        output = self.output_fc(transformer_output)
        return output

class GatingNetwork(nn.Module):
    def __init__(self, input_feature_dim, num_experts, hidden_dims=None, dropout_rate=0.0):
        super(GatingNetwork, self).__init__()
        layers = []
        last_dim = input_feature_dim
        

        if hidden_dims is not None:
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(last_dim, hidden_dim))
                layers.append(nn.ReLU())
                if dropout_rate > 0.0:
                    layers.append(nn.Dropout(dropout_rate))
                last_dim = hidden_dim
        
        layers.append(nn.Linear(last_dim, num_experts))
        self.fc_layers = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = x.mean(dim=1)
        x = self.fc_layers(x)
        return self.softmax(x)

class MixtureOfTransformerExperts(nn.Module):
    def __init__(self, input_size, d_model, output_size, nhead, dim_feedforward, num_experts, num_encoder_layers=1):
        super(MixtureOfTransformerExperts, self).__init__()
        self.num_experts = num_experts
        self.output_size = output_size
        self.experts = nn.ModuleList([TransformerExpert(input_size, d_model, output_size, nhead, dim_feedforward, num_encoder_layers) for _ in range(num_experts)])
        self.gating_network = GatingNetwork(d_model, num_experts)

    def forward(self, x):
        gating_scores = self.gating_network(x)

        expert_outputs = [expert(x) for expert in self.experts]
        stacked_expert_outputs = torch.stack(expert_outputs)

        expanded_gating_scores = gating_scores.unsqueeze(2).unsqueeze(3)
        expanded_gating_scores = expanded_gating_scores.expand(-1, -1, x.size(1), self.output_size)
        expanded_gating_scores = expanded_gating_scores.transpose(0, 1)

        mixed_output = torch.sum(stacked_expert_outputs * expanded_gating_scores, dim=0)
        return mixed_output

class MoETransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, moe):
        super(MoETransformerModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.moe = moe
        self.dropout = nn.Dropout(p=0.125)

    def forward(self, x, teacher_inputs=None):  # Updated forward method
        embedded = self.dropout(self.embedding(x))
        
        if teacher_inputs is not None: # Using teacher forcing
            return self.moe(embedded) 
        else:
            return self.moe(embedded) 

# ---------- Dataset Definitions ----------
class QAJsonlDataset(Dataset):
    def __init__(self, path, seq_len):
        self.seq_len = seq_len
        self.pairs = self.load_data(path)
        self.vocab, self.idx2token = self.build_vocab([word for pair in self.pairs for sublist in pair for word in sublist])
        self.tokenized_pairs = [(self.tokenize(q), self.tokenize(a)) for q, a in self.pairs]

    def load_data(self, path):
        pairs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                question, answer = data.get("question", ""), data.get("answer", "")
                pairs.append((question.split(), answer.split()))
        return pairs

    def tokenize(self, words):
        tokens = [self.vocab.get(w, self.vocab["<unk>"]) for w in words]
        if len(tokens) < self.seq_len:
            tokens.append(self.vocab["<eos>"])
            tokens.extend([self.vocab["<pad>"]] * (self.seq_len - len(tokens)))
        else:
            tokens = tokens[:self.seq_len - 1] + [self.vocab["<eos>"]]
        return tokens

    def build_vocab(self, words):
        vocab = {"<unk>": 0, "<pad>": 1, "<eos>": 2}
        start_index = len(vocab)
        counts = Counter(words)
    
        for word, _ in counts.most_common():
            if word not in vocab:
                vocab[word] = len(vocab)

        idx2token = {idx: token for token, idx in vocab.items()}
        return vocab, idx2token
        
    def __len__(self):
        return len(self.tokenized_pairs)

    def __getitem__(self, idx):
        tokenized_question, tokenized_answer = self.tokenized_pairs[idx]
        return torch.tensor(tokenized_question, dtype=torch.long), torch.tensor(tokenized_answer, dtype=torch.long)

def collate_fn(batch):
    questions, answers = zip(*batch)
    questions = pad_sequence(questions, batch_first=True, padding_value=0)
    answers = pad_sequence(answers, batch_first=True, padding_value=0)
    return questions, answers

# ---------- Training and Inference Functions ----------
def train_model(model, criterion, optimizer, scheduler, num_epochs, train_loader, val_loader, patience=5, teacher_forcing_ratio=0.5):
    model.train()

    best_val_loss = float('inf')
    epochs_without_improvement = 0 

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for i, (inputs, targets) in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            use_teacher_forcing = True if torch.rand(1).item() < teacher_forcing_ratio else False

            if use_teacher_forcing:
                teacher_inputs = torch.cat((torch.tensor([[2]] * inputs.size(0)).to(device), targets[:, :-1]), dim=1) 
                predictions = model(inputs, teacher_inputs) 
            else:
                predictions = model(inputs) 

            mask = (targets != 0)  
            predictions = predictions.view(-1, predictions.size(-1))
            targets = targets.view(-1)

            # Apply mask directly to the loss calculation
            loss = criterion(predictions[mask], targets[mask])  # Corrected line  # Corrected line 

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"}) # Improved output format

        average_loss = total_loss / len(train_loader.dataset)
        train_losses.append(average_loss)
        print(f"Epoch {epoch+1}, Average Loss: {average_loss}")

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_predictions = model(val_inputs)
                
                val_predictions = val_predictions.view(-1, val_predictions.size(-1))
                val_targets = val_targets.view(-1)
                val_loss += criterion(val_predictions, val_targets, mask=mask).item() 
        average_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(average_val_loss)  # Append average validation loss
        print(f"Validation Loss: {average_val_loss}")

        # Early stopping check
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            epochs_without_improvement = 0
            # You might want to save the best model here
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping activated after {epoch+1} epochs.")
                break

        model.train() 
        scheduler.step(average_val_loss)  # Update learning rate scheduler

    # Visualize training and validation curves
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def generate_text_beam_search(model, dataset, seed_text, max_length, beam_width=3, temperature=1.0):
    model.eval()
    end_token_id = dataset.vocab["<eos>"]
    
    # Initial input
    input_sequence = [dataset.vocab.get(word, dataset.vocab["<pad>"]) for word in seed_text.split()]
    input_sequence = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0).to(device)

    # Beam search initialization
    beam = [(input_sequence, 0.0)]  # (sequence, log_probability)

    for _ in range(max_length):
        new_beam = []
        for sequence, log_prob in beam:
            with torch.no_grad():
                output = model(sequence)
            
            # Get probabilities for the last token
            probabilities = F.softmax(output[:, -1, :] / temperature, dim=-1)
            
            # Get top k probabilities and their indices
            topk_probs, topk_ids = torch.topk(probabilities, beam_width, dim=-1)

            for prob, token_id in zip(topk_probs[0], topk_ids[0]):
                new_sequence = torch.cat([sequence, token_id.unsqueeze(0).unsqueeze(0)], dim=1)
                new_log_prob = log_prob + torch.log(prob).item() 
                new_beam.append((new_sequence, new_log_prob)) 

        # Sort by probability and select top k
        beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_width] 

        # If end token is generated in the most likely sequence, stop
        if beam[0][0][0][-1].item() == end_token_id: 
            break 

    # Get the best sequence and convert it to text
    best_sequence = beam[0][0][0].tolist()
    generated_text = " ".join([dataset.idx2token.get(token, "<unk>") for token in best_sequence])
    return generated_text

# Function to calculate BLEU score
def calculate_bleu(reference, candidate):
    smoothing_function = SmoothingFunction().method4
    bleu_score = sentence_bleu(
        [reference.split()], candidate.split(), smoothing_function=smoothing_function
    )
    return bleu_score

# Function to calculate ROUGE scores
def calculate_rouge(reference, candidate):
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)
    return scores[0]

def generate_text(model, dataset, seed_text, num_generate, temperature=1.0):
    model.eval()
    
    generated_tokens = []

    # Initial sequence (prefix) to start the generation process
    input_sequence = [dataset.vocab.get(word, dataset.vocab["<pad>"]) for word in seed_text.split()]
    current_sequence = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0)
    current_sequence = current_sequence.to(device)

    # Generate num_generate tokens
    for _ in range(num_generate):
        # Forward pass through the model
        with torch.no_grad():
            output = model(current_sequence)
        
        # Get probabilities, apply temperature scaling, and sample from the distribution
        probabilities = F.softmax(output[:, -1, :] / temperature, dim=-1).detach()
        next_token_idx = torch.multinomial(probabilities, 1).item()

        # Append token to the current sequence and to the generated tokens
        generated_tokens.append(next_token_idx)
        current_sequence = torch.cat((current_sequence, torch.tensor([[next_token_idx]])), 1).to(device)
    
    # Convert tokens to words
    generated_text = " ".join([dataset.idx2token.get(token, "<unk>") for token in generated_tokens])
    return generated_text

def count_tokens_in_dataset(dataset):
    return sum([len(pair[0]) + len(pair[1]) for pair in dataset.pairs])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ---------- Hyperparameters and Model Instantiation ----------
# Transformer :
d_model = 384
nhead = 8
dim_feedforward = 768
num_encoder_layers = 8
num_experts = 2
hidden_dims = [512, 256]
dropout_rate = 0.5

# Gating Network :
gating_network = GatingNetwork(
    input_feature_dim=d_model,
    num_experts=num_experts,
    hidden_dims=hidden_dims,
    dropout_rate=dropout_rate,
)

# Dataset :
path_to_dataset = "train.jsonl"
seq_len = 24
dataset = QAJsonlDataset(path_to_dataset, seq_len)

# Split the dataset into training and validation sets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=seq_len, shuffle=True, collate_fn=collate_fn, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=seq_len, shuffle=False, collate_fn=collate_fn, pin_memory=True)
    
num_tokens = count_tokens_in_dataset(dataset)
print(f"Total number of tokens in the dataset: {num_tokens}")

vocab_size = len(dataset.vocab)

moe = MixtureOfTransformerExperts(
    input_size=d_model,
    d_model=d_model,
    output_size=vocab_size,
    nhead=nhead,
    dim_feedforward=dim_feedforward,
    num_experts=num_experts,
    num_encoder_layers=num_encoder_layers
).to(device)

moe_transformer_model = MoETransformerModel(vocab_size, d_model, moe).to(device)

# Count of total parameters :
total_params = count_parameters(moe_transformer_model)
print(f"Total trainable parameters: {total_params}")

# ---------- Training ----------
num_epochs = 10 
learning_rate = 1e-4
teacher_forcing_ratio = 0.5 # Initial teacher forcing ratio

criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab["<pad>"])
optimizer = torch.optim.AdamW(moe_transformer_model.parameters(), lr=learning_rate, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

# Train the model 
train_model(moe_transformer_model, criterion, optimizer, scheduler, num_epochs, train_loader, val_loader, teacher_forcing_ratio=teacher_forcing_ratio)


# ---------- Inference ----------
def interactive_text_generation(model, dataset, max_length=20, temperature=1.0):
    while True:
        try:
            seed_text = input("Enter seed text (type 'quit' to exit and save the model): ").strip()
            if seed_text.lower() == 'quit':
                print("Exiting text generation mode.")
                break
            if seed_text:
                # Use beam search for text generation
                generated_text = generate_text_beam_search(model, dataset, seed_text, max_length, temperature=temperature)
                print("Generated Text (Beam Search): ", generated_text)

                # For evaluation, you'll need a reference text
                reference_text = input("Enter a reference text for evaluation: ").strip()  
                if reference_text:
                    bleu_score = calculate_bleu(reference_text, generated_text)
                    rouge_scores = calculate_rouge(reference_text, generated_text)

                    print(f"BLEU Score: {bleu_score:.4f}")
                    print("ROUGE Scores:")
                    for metric, score in rouge_scores.items():
                        print(f"- {metric}: {score}")
            else:
                print("Seed text cannot be empty. Please enter some text.")
        except KeyboardInterrupt:
            print("\nReceived interrupt signal. Exiting text generation mode.")
            break
        except Exception as e:
            print(f"An error occurred: {e}. Try again.")

interactive_text_generation(moe_transformer_model, dataset)

# ---------- Save Trained Model ----------
torch.save(moe_transformer_model.state_dict(), "MoE_Transformer.pth")
