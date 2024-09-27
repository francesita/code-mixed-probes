# -*- coding: utf-8 -*-
"""
All this code is derived from [this tutorial on how to train STS model using BERT](https://thepythoncode.com/article/finetune-bert-for-semantic-textual-similarity-in-python)
"""

from sentence_transformers import SentenceTransformer, models
from transformers import get_linear_schedule_with_warmup, BertTokenizer, XLMRobertaTokenizer
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import datetime
import random
import pandas as pd


def get_tokenizer(encoder):
    if encoder.split('-')[0] == 'bert':
        tokenizer = BertTokenizer.from_pretrained(encoder)
    else:
        tokenizer = XLMRobertaTokenizer.from_pretrained(encoder)

    return tokenizer


class STSBDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, encoder):
        # normalise similarity scores in dataset
        similarity_scores = [i['similarity_score'] for i in dataset]
        self.normalized_sim_scores = [i / 5.0 for i in similarity_scores]
        self.first_sentences = [i['sentence1'] for i in dataset]
        self.second_sentences = [i['sentence2'] for i in dataset]
        self.concatenated_sentences = [[str(x), str(y)] for x, y in zip(self.first_sentences, self.second_sentences)]
        self.tokenizer = get_tokenizer(encoder)

    def __len__(self):
        return len(self.concatenated_sentences)

    def get_batch_labels(self, idx):
        return torch.tensor(self.normalized_sim_scores[idx])

    def get_batch_texts(self, idx):
        return self.tokenizer(self.concatenated_sentences[idx], padding='max_length', max_length=128, truncation=True,
                              return_tensors="pt")

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


def collate_fn(texts):
    input_ids = texts['input_ids']
    attention_masks = texts['attention_mask']
    features = [{'input_ids': input_id,
                 'attention_mask': attention_mask} for input_id, attention_mask in
                zip(input_ids, attention_masks)]
    return features


class TransformerForSTS(torch.nn.Module):

    def __init__(self, encoder):
        super(TransformerForSTS, self).__init__()
        self.encoder = models.Transformer(encoder, max_seq_length=128)
        self.pooling_layer = models.Pooling(self.encoder.get_word_embedding_dimension())
        self.sts_encoder = SentenceTransformer(modules=[self.encoder, self.pooling_layer])

    def forward(self, input_data):
        output = self.sts_encoder(input_data)['sentence_embedding']
        return output


"""According to tutorial and very logically, we wish to differentiate between similar and dissimilar sentences. We want the model to assign dissimilar pairs a large distance while keeping texts close in meaning a small distance to each other, or assign a small similarity score. Cosine similarity loss should be used to accomplish this.

This makes sense when we think about traditional static word embeddings and how cosine similarity is used to tell which words are similar and dissimilar.
"""


class CosineSimilarityLoss(torch.nn.Module):

    def __init__(self, loss_fn=torch.nn.MSELoss(), transform_fn=torch.nn.Identity()):
        super(CosineSimilarityLoss, self).__init__()
        self.loss_fn = loss_fn
        self.transform_fn = transform_fn
        self.cos_similarity = torch.nn.CosineSimilarity(dim=1)

    def forward(self, inputs, labels):
        emb_1 = torch.stack([inp[0] for inp in inputs])
        emb_2 = torch.stack([inp[1] for inp in inputs])
        outputs = self.transform_fn(self.cos_similarity(emb_1, emb_2))
        return self.loss_fn(outputs, labels.squeeze())


# Takes a time in seconds and returns a string hh:mm:ss
def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train(model, epochs, train_dataloader, val_dataloader, optimizer, scheduler, device):
    seed_val = 42

    criterion = CosineSimilarityLoss()
    criterion = criterion.to(device)

    random.seed(seed_val)
    torch.manual_seed(seed_val)

    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []
    total_t0 = time.time()

    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()

        total_train_loss = 0

        model.train()

        # For each batch of training data...
        for train_data, train_label in tqdm(train_dataloader):
            train_data['input_ids'] = train_data['input_ids'].to(device)
            train_data['attention_mask'] = train_data['attention_mask'].to(device)

            train_data = collate_fn(train_data)
            model.zero_grad()

            output = [model(feature) for feature in train_data]

            loss = criterion(output, train_label.to(device))
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.5f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")

        t0 = time.time()

        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for val_data, val_label in tqdm(val_dataloader):
            val_data['input_ids'] = val_data['input_ids'].to(device)
            val_data['attention_mask'] = val_data['attention_mask'].to(device)

            val_data = collate_fn(val_data)

            with torch.no_grad():
                output = [model(feature) for feature in val_data]

            loss = criterion(output, val_label.to(device))
            total_eval_loss += loss.item()

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(val_dataloader)

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.5f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    return model, training_stats


def predict_similarity(sentence_pair, tokenizer, model, device):
    test_input = tokenizer(sentence_pair, padding='max_length', max_length=128, truncation=True,
                           return_tensors="pt").to(device)
    test_input['input_ids'] = test_input['input_ids']
    test_input['attention_mask'] = test_input['attention_mask']
    # del test_input['token_type_ids']
    output = model(test_input)
    sim = torch.nn.functional.cosine_similarity(output[0], output[1], dim=0).item()

    return sim


def load_model(encoder, model_path, device):
    # load model
    model = TransformerForSTS(encoder)
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model
