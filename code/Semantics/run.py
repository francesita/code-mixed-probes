import logging
import math

import torch
import yaml
import os
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, models, util, InputExample
from transformers import get_linear_schedule_with_warmup, AdamW
import semantictextsimilarity
from semantictextsimilarity import TransformerForSTS
from datasets import load_dataset, concatenate_datasets
from sentence_transformers import evaluation, SentenceTransformer
from scipy import stats
import matplotlib.pyplot as plt

with open('config.yaml', 'r') as f:
    args = yaml.safe_load(f)

train_model = args['train_model']
# encoder = args['model']['name']
results_save_path = os.path.join(args['reporting']['root'], args['reporting']['csv_path'])
seeds = args['training_params']['seeds']
device = torch.device("cuda")


# tokenizer = semantictextsimilarity.get_tokenizer(encoder)


def prep_data(dataset, norm=True):
    first_sent = [i['sentence1'] for i in dataset]
    second_sent = [i['sentence2'] for i in dataset]
    if norm:
        norm_labels = [i['similarity_score'] / 5.0 for i in dataset]
    else:
        norm_labels = [i['similarity_score'] for i in dataset]
    # turn test dataset intp InputExample type [InputExample(texts=[sentence1, sentence2], label=0.8), InputExampe([
    # sen1, sen2], label=0.2)]
    return [InputExample(texts=[str(x), str(y)], label=float(z)) for x, y, z in
            zip(first_sent, second_sent, norm_labels)]


def plot_df(df, encoder, spearman_rank, test):
    x = df.preds
    y = df.gold_labels
    plt.scatter(x, y)
    plt.plot(x.sort_values(), y.sort_values(), color='red')
    plt.title("Spearman correlation coefficient: {:.2f}".format(spearman_rank.statistic))
    plt.xlabel("predictions")
    plt.ylabel("gold labels")
    plt.savefig(results_save_path + f"spearman_rank_test_{encoder}_{test}")
    plt.show()


def test_cs_experiments(list_df, lang1, lang2, lr=args['training_params']['learning_rate']):
    list_preds = []
    for i, df in enumerate(list_df):
        first_sent = [df['sentence1'][j] for j in range(len(df))]
        len(first_sent)
        second_sent = [df['sentence2'][j] for j in range(len(df))]

        first_sent = model.encode(first_sent, convert_to_tensor=True)
        second_sent = model.encode(second_sent, convert_to_tensor=True)
        pred_scores = util.cos_sim(first_sent, second_sent).detach().cpu().numpy().tolist()
        pred_scores = [pred_scores[k][k] for k in range(len(first_sent))]
        print(type(pred_scores), pred_scores)
        list_preds.append(pred_scores)

    # Once we have two pred scores (one for each data we pair against each other)
    # get spearman rank correlation
    print(list_preds[0], print(len(list_preds[0])))
    print('length of preds list is', len(list_preds))
    preds = list_preds[0]
    gold = list_preds[1]  # not real gold, just other preds cuz we compare against each other
    spearman_rank = stats.spearmanr(preds, gold)
    print(spearman_rank.statistic)

    outputs_df = pd.DataFrame({'preds': preds,
                               'gold_labels': gold})

    outputs_df.to_csv(os.path.join(results_save_path, f'sts_model_{encoder}_{lang1}_{lang2}_{lr}_preds.csv'))
    plot_df(outputs_df, encoder, spearman_rank, f'{lang1}_vs_{lang2}')

    # to get score from evaluator using sentbert library
    # 1. create dataset in right format
    # add pred scores to original df so that we can use later of sentbert eval
    list_df[0]['scores'] = list_preds[1]    # we assing the opposite pred scores as gold labels b/c we want to compare model output for this
    list_df[1]['scores'] = list_preds[0]

    # prep data
    first_sent = [df['sentence1'][j] for j in range(len(list_df[0]))]
    second_sent = [df['sentence2'][j] for j in range(len(list_df[0]))]
    norm_labels = [df['scores'][j] for j in range(len(list_df[1]))]   # opposite scores b/c we compare model gen against this
    examples = [InputExample(texts=[str(x), str(y)], label=float(z)) for x, y, z in
            zip(first_sent, second_sent, norm_labels)]

    # 2. define test evaluator
    test_evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(examples, name=f'sts-{lang1}_{lang2}_{lr}')
    test_evaluator(model, output_path=results_save_path)


def test_after_train(model, encoder, lr=args['training_params']['learning_rate']):
    # test partition of train sts dataset/
    en_dataset = load_dataset("stsb_multi_mt", "en")
    es_dataset = load_dataset("stsb_multi_mt", "es")

    print(en_dataset)
    print(es_dataset)

    print(es_dataset['train'][100])
    print(en_dataset['train'][100])

    test_dataset = concatenate_datasets([es_dataset['test'], en_dataset['test']])
    test_examples = prep_data(test_dataset)

    test_evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(test_examples, name=f'sts-{lr}')
    test_evaluator(model, output_path=results_save_path)

    #### My eval method
    first_sent = [i['sentence1'] for i in test_dataset]
    second_sent = [i['sentence2'] for i in test_dataset]
    norm_labels = [i['similarity_score'] / 5.0 for i in test_dataset]
    # turn test dataset intp InputExample type [InputExample(texts=[sentence1, sentence2], label=0.8), InputExampe([
    # sen1, sen2], label=0.2)]
    # full_text = [[str(x), str(y)] for x, y in zip(first_sent, second_sent)]

    first_sent = model.encode(first_sent, convert_to_tensor=True)
    second_sent = model.encode(second_sent, convert_to_tensor=True)
    pred_scores = util.cos_sim(first_sent, second_sent).detach().cpu().numpy().tolist()
    sim_scores = [score for score in norm_labels]

    preds = [pred_scores[i][i] for i in range(len(first_sent))]
    print(len(preds))
    print(len(sim_scores))

    # get spearman rank correlation
    spearman_rank = stats.spearmanr(preds, sim_scores)
    print(spearman_rank.statistic)

    outputs_df = pd.DataFrame({'preds': preds,
                               'gold_labels': sim_scores})

    outputs_df.to_csv(os.path.join(results_save_path, f'sts_model_{encoder}_{lr}_preds.csv'))
    plot_df(outputs_df, encoder, spearman_rank)



if train_model:
    en_dataset = load_dataset("stsb_multi_mt", "en")
    es_dataset = load_dataset("stsb_multi_mt", "es")

    print(en_dataset)
    print(es_dataset)

    print(es_dataset['train'][100])
    print(en_dataset['train'][100])

    train_dataset = concatenate_datasets([es_dataset['train'], en_dataset['train']])
    dev_dataset = concatenate_datasets([es_dataset['dev'], en_dataset['dev']])
    test_dataset = concatenate_datasets([es_dataset['test'], en_dataset['test']])

    print(train_dataset)
    print(train_dataset[500])

    """## Prepare training and validation data split"""

    # if we use fit function to train, it tokenizes our data for us
    train_ds = prep_data(train_dataset)
    val_ds = prep_data(dev_dataset)

    # we split 90/10
    train_size = len(train_ds)
    val_size = len(val_ds)

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))

    epochs = int(args['training_params']['epochs'])
    batch_size = int(args['training_params']['batch_size'])

    train_dataloader = DataLoader(
        train_ds,  # The training samples.
        num_workers=4,
        batch_size=batch_size,  # Use this batch size.
        shuffle=True  # Select samples randomly for each batch
    )

    val_dataloader = DataLoader(
        val_ds,
        num_workers=4,
        batch_size=batch_size  # Use the same batch size
    )

   
    encoders = ['xlm-roberta-large']

    for encoder in encoders:
        # make model
        word_embedding_model = models.Transformer(encoder)

        # Apply mean pooling to get one fixed sized sentence vector
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)

        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        train_loss = losses.CosineSimilarityLoss(model=model)
        evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(val_ds,
                                                                                name='sts-dev')  # testing the model

        #
        # evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(test_examples, name=f'{encoder}-sts-2e-5-lr')
        # evaluator(model, output_path=results_save_path)

        # Configure the training. We skip evaluation in this example
        warmup_steps = math.ceil(len(train_dataloader) * epochs * 0.1)  # 10% of train data for warm-up
        logging.info("Warmup-steps: {}".format(warmup_steps))

        model_path = os.path.join(args['training_params']['root'], args['training_params']['save_path'],
                                  f'{encoder}_sts_fit_2e-6.params')

        model.fit(train_objectives=[(train_dataloader, train_loss)],
                  evaluator=evaluator,
                  epochs=epochs,
                  optimizer_params={'lr': float(args['training_params']['learning_rate'])},
                  evaluation_steps=1000,
                  warmup_steps=warmup_steps,
                  output_path=model_path)

        # model = SentenceTransformer(model_path)
        test_examples = prep_data(test_dataset)
        test_evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(test_examples,
                                                                                     name=f'{encoder}-sts-2e-5-lr')
        test_evaluator(model, output_path=results_save_path)

else:

    # cs experiment examples
    data_root = args['dataset']['root']

    cs_cs_df = pd.read_csv(os.path.join(data_root, 'randcs_randcs_sts_pair.csv'))
    es_es_df = pd.read_csv(os.path.join(data_root, 'es_es_sts_pair.csv'))
    en_en_df = pd.read_csv(os.path.join(data_root, 'en_en_sts_pair.csv'))
    en_es_df = pd.read_csv(os.path.join(data_root, 'es_en_sts_pairs.csv'))
    cs_es_df = pd.read_csv(os.path.join(data_root, 'randcs_es_sts_pairs.csv'))
    cs_en_df = pd.read_csv(os.path.join(data_root, 'randcs_en_sts_pairs.csv'))

    # shuffle data in dataframes using the same seed
    cs_cs = cs_cs_df.sample(frac=1, random_state=42).reset_index(drop=True)[:6000]
    es_es = es_es_df.sample(frac=1, random_state=42).reset_index(drop=True)[:6000]
    en_en = en_en_df.sample(frac=1, random_state=42).reset_index(drop=True)[:6000]
    en_es = en_es_df.sample(frac=1, random_state=42).reset_index(drop=True)[:6000]
    cs_es = cs_es_df.sample(frac=1, random_state=42).reset_index(drop=True)[:6000]
    cs_en = cs_en_df.sample(frac=1, random_state=42).reset_index(drop=True)[:6000]

    print(cs_cs[:100])
    print(es_es.head())
    print(en_en.head())

    encoders = ['bert-base-multilingual-uncased', 'xlm-roberta-base']  # ,['xlm-roberta-large']  #
    for encoder in encoders:
        model_path = os.path.join(args['training_params']['root'], args['training_params']['save_path'],
                                  f'{encoder}_sts_fit_2e-5.params')
        model = SentenceTransformer(model_path)
        test_cs_experiments([cs_cs, es_es], 'cs_cs', 'es_es', lr=args['training_params']['learning_rate'])
        test_cs_experiments([cs_cs, en_en], 'cs_cs', 'en_en', lr=args['training_params']['learning_rate'])

        test_cs_experiments([en_es, cs_es], 'en_es', f'cs_es{encoder}', lr=args['training_params']['learning_rate'])
        test_cs_experiments([en_es, cs_en], 'en_es', f'cs_en{encoder}', lr=args['training_params']['learning_rate'])
