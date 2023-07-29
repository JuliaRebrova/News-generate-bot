import os
from transformers import GPT2LMHeadModel, AdamW
from prepare_dataset import Dataset, Tokenizer

import numpy as np

import torch
import transformers

if torch.cuda.is_available():    
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


BATCH_SIZE = 8
MAX_LEN = 70
EPOCHS = 5
DATA_PATH = os.path.join("..", "data")
MODEL_PATH = os.path.join("..", "models", "model.pth")


def accuracy(y_true, logits):
    return torch.mean((y_true[1:] == torch.argmax(logits, dim=2)[:-1]).float()).detach().cpu().numpy()


def prep_tensors(x, i, batch_size=BATCH_SIZE, max_len=MAX_LEN):
    batch_ids = x[i*batch_size*max_len: (i+1)*batch_size*max_len]
    batch_ids = batch_ids.reshape(batch_size, max_len)
    batch_ids = torch.tensor(batch_ids).to(device)
    return batch_ids

def train(model, epochs, train, test, batch_size, max_len, optimizer):
    n_train = len(train)//(batch_size * max_len)
    n_test = len(test)//(batch_size * max_len)
    total_steps = n_train * EPOCHS
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    for epoch in range(1, epochs+1):
        print(f'epoch {epoch}/{epochs} : training')

        train_loss = []
        train_acc = []
        model.train()
        # pbar = range(n_train)
        for i in range(n_train):
            batch_ids = prep_tensors(train, i)

            model.zero_grad()
            loss, logits, _ = model(batch_ids,
                                token_type_ids=None, 
                                #  attention_mask=batch_mask,
                                labels=batch_ids
                                ).values()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss.append(loss.item())
            train_acc.append(accuracy(batch_ids, logits))
            print(f'acc {np.mean(train_acc):.4f} loss {np.mean(train_loss):.4f}')

        
        print(f'epoch {epoch}/{epochs} : validation')
        model.eval()
        val_acc = []
        val_loss = []
        # pbar = tqdm(range(n_test))
        for i in range(n_test):
            batch_ids = prep_tensors(test, i)
            with torch.no_grad():        
                loss, logits, _ = model(batch_ids, 
                                    token_type_ids=None, 
                                    # attention_mask=batch_mask,
                                    labels=batch_ids
                                    ).values()
            
            val_loss.append(loss.item())
            val_acc.append(accuracy(batch_ids, logits))
            print(f'acc {np.mean(val_acc):.4f} loss {np.mean(val_loss):.4f}')
    torch.save(model, MODEL_PATH)




if __name__ == "__main__":
    model = GPT2LMHeadModel.from_pretrained(
        'sberbank-ai/rugpt3small_based_on_gpt2',
        output_attentions = False,
        output_hidden_states = False,
    )
    model.to(device)
    optimizer = AdamW(model.parameters(), lr = 1e-5, eps = 1e-8)

    row_data = Dataset(DATA_PATH).texts
    train_data, test_data = Tokenizer().tokenize_train_test(row_data)
    
    train(model, EPOCHS, train_data, test_data, BATCH_SIZE, MAX_LEN, optimizer)
