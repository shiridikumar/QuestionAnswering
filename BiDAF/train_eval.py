import torch
import torch.nn as nn
import torch.nn.functional as F
import json

def train(model, train_dataset, optimizer, device):
    print("Commencing training...")
    total_loss = 0.0
    model.train()

    for batch_count, batch in enumerate(train_dataset):
        optimizer.zero_grad()
        if batch_count % 500 == 0:
            print(f"Processing batch: {batch_count}")

        context, question, char_ctx, char_ques, label, _, _, _ = map(lambda x: x.to(device), batch)
        preds = model(context, question, char_ctx, char_ques)

        start_pred, end_pred = preds
        s_idx, e_idx = label[:, 0], label[:, 1]

        loss = F.cross_entropy(start_pred, s_idx) + F.cross_entropy(end_pred, e_idx)
        loss.backward()
        plot_grad_flow(model.named_parameters())
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_dataset)

def validate(model, valid_dataset, device, idx2word):
    print("Initiating validation...")
    total_loss, f1, em = 0.0, 0.0, 0.0
    model.eval()

    predictions = {}

    for batch_count, batch in enumerate(valid_dataset):
        if batch_count % 500 == 0:
            print(f"Processing batch: {batch_count}")

        context, question, char_ctx, char_ques, label, _, _, ids = map(lambda x: x.to(device), batch)
        with torch.no_grad():
            s_idx, e_idx = label[:, 0], label[:, 1]
            preds = model(context, question, char_ctx, char_ques)
            p1, p2 = preds

            loss = F.cross_entropy(p1, s_idx) + F.cross_entropy(p2, e_idx)
            total_loss += loss.item()

            batch_size, c_len = p1.size()
            ls = nn.LogSoftmax(dim=1)

            mask = (torch.ones(c_len, c_len) * float('-inf')).to(device).tril(-1).unsqueeze(0).expand(batch_size, -1, -1)
            score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
            score, s_idx = score.max(dim=1)
            score, e_idx = score.max(dim=1)
            s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()

            for i in range(batch_size):
                current_id = ids[i]
                pred = ' '.join([idx2word[idx.item()] for idx in context[i][s_idx[i]:e_idx[i]+1]])
                predictions[current_id] = pred

    em, f1 = evaluate(predictions)
    return total_loss / len(valid_dataset), em, f1

def evaluate(predictions):
    with open('./data/squad_dev.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    dataset = dataset['data']

    f1, em, total = 0, 0, 0

    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                exact_match = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
                em += exact_match

    em = 100.0 * em / total
    f1 = 100.0 * f1 / total
    return em, f1
##example
# for epoch in range(epochs):
#     print(f"Epoch {epoch+1}")
#     start_time = time.time()
#     train_loss = train(model, train_dataset)
#     valid_loss, em, f1 = valid(model, valid_dataset)
#     torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': valid_loss,
#             'em':em,
#             'f1':f1,
#             }, 'bidaf_run4_{}.pth'.format(epoch))
#     end_time = time.time()
#     epoch_mins, epoch_secs = epoch_time(start_time, end_time)
#     train_losses.append(train_loss)
#     valid_losses.append(valid_loss)
#     ems.append(em)
#     f1s.append(f1)
#     print(f"Epoch train loss : {train_loss}| Time: {epoch_mins}m {epoch_secs}s")
#     print(f"Epoch valid loss: {valid_loss}")
#     print(f"Epoch EM: {em}")
#     print(f"Epoch F1: {f1}")
#     print("====================================================================================")
