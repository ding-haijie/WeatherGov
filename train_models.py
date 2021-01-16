import datetime
import math
import time
import torch.nn as nn
from torch import optim, Tensor

from data_processor import DataProcessor
from utils import *
from evals import BleuScore
from models import EncoderGate, DecoderAttn, Data2Text
from config import params

max_len = params["max_len"]
input_dim = params["input_dim"]
output_dim = params["output_dim"]
batch_size = params["batch_size"]
max_epoch = params["epochs"]
learning_rate = params["learning_rate"]
embed_dim = params["embed_dim"]
hidden_dim = params["hidden_dim"]
dropout = params["dropout"]
weight_decay = params["weight_decay"]
gamma = params["gamma"]
clip = params["clip"]
seed = params["seed"]
beam_width = params["beam_width"]
resume = params['resume']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fix_seed(seed)

SOS_TOKEN, EOS_TOKEN = 0, 1

check_file_exist('./results')
check_file_exist('./results/logs')
cur_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M')
logger = get_logger('./results/logs/' + cur_time + '.log')
logger.info(f'GPU_device: {torch.cuda.get_device_name()}')

start_time = time.time()
data_processor = DataProcessor()

train_data_loader = get_data_loader(data_processor, batch_size, device, 'train')
dev_data_loader = get_data_loader(data_processor, batch_size, device, 'dev')

end_process_data = time.time()
logger.info(f'data processing consumes: {(time.time() - start_time):.2f}s')
for item in params:
    logger.info(f'{item}: {params[item]}')


def weights_init(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.kaiming_normal_(param.data, mode='fan_in', nonlinearity='relu')
        else:
            nn.init.constant_(param.data, 0)


encoder = EncoderGate(input_dim, embed_dim, hidden_dim, dropout)
decoder = DecoderAttn(output_dim, embed_dim, hidden_dim, dropout)
model = Data2Text(encoder, decoder, beam_width).to(device)
model.apply(weights_init)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.NLLLoss()

# load checkpoint and continue training
if resume:
    checkpoint = load_checkpoint(latest=True)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])


# noinspection DuplicatedCode
def train(data_loader):
    model.train()
    epoch_loss = 0
    for train_input, train_target in data_loader:
        train_input = train_input.permute(1, 0, 2)
        train_target = train_target.permute(1, 0, 2)
        optimizer.zero_grad()
        train_output, content_selection = model(train_input, train_target, train_mode=True)
        train_output = train_output[:-1].view(-1, train_output.size(-1))

        # custom loss function
        loss = criterion(train_output, train_target[1:].reshape(-1))
        gamma_tensor = torch.tensor([gamma for _ in range(train_target.size(1))], device=device)
        reg_term = torch.pow((torch.sum(content_selection, dim=0).squeeze(dim=-1) - gamma_tensor), 2) + (
                torch.tensor([1.0 for _ in range(train_target.size(1))], device=device) -
                torch.max(content_selection, dim=0)[0])
        loss += reg_term.mean()

        loss.backward()
        # avoid gradient exploding
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(data_loader)


# noinspection DuplicatedCode
def validate(data_loader):
    model.eval()
    epoch_loss = 0
    epoch_perplexity = 0
    with torch.no_grad():
        for validate_input, validate_target in data_loader:
            validate_input = validate_input.permute(1, 0, 2)
            validate_target = validate_target.permute(1, 0, 2)
            validate_output, content_selection = model(
                validate_input, validate_target, train_mode=True)

            # truncate actual output and target (remove [SOS]/[EOS])
            validate_output, validate_target = validate_output[:-1], validate_target[1:]
            # calculate perplexity: perplexity = p(w1, w2 ... wN) ** (-1 / N)
            for batch_idx in range(validate_output.size(1)):
                batch_prob = 1.0
                for seq_idx in range(validate_output.size(0)):
                    prob = validate_output[seq_idx][batch_idx][validate_target[seq_idx][batch_idx][0].item()].item()
                    batch_prob *= math.exp(prob)
                batch_prob = batch_prob ** (-1 / validate_output.size(0))
                epoch_perplexity += batch_prob
            epoch_perplexity /= validate_output.size(1)

            # custom loss function
            validate_output = validate_output.view(-1, validate_output.size(-1))
            loss = criterion(validate_output, validate_target.reshape(-1))
            gamma_tensor = torch.tensor([gamma for _ in range(validate_target.size(1))], device=device)
            reg_term = torch.pow((torch.sum(content_selection, dim=0).squeeze(dim=-1) - gamma_tensor), 2) + (
                    torch.tensor([1.0 for _ in range(validate_target.size(1))], device=device) -
                    torch.max(content_selection, dim=0)[0])
            loss += reg_term.mean()
            epoch_loss += loss.item()

    return epoch_loss / len(data_loader), epoch_perplexity / len(data_loader)


def evaluate(inference_input: Tensor):
    model.eval()
    with torch.no_grad():
        inference_input = inference_input.permute(1, 0, 2)
        inference_target = torch.tensor([[SOS_TOKEN]], dtype=torch.long, device=device)
        inference_output, _ = model(inference_input, inference_target, train_mode=False)
        # inference_align = ((content_selection.squeeze() > 0.5) * 1).nonzero()

    return inference_output


loss_dict_train, loss_dict_dev = [], []
for epoch in range(1, max_epoch + 1):
    start_time = time.time()
    train_loss = train(train_data_loader)
    dev_loss, dev_ppl = validate(dev_data_loader)
    loss_dict_train.append(train_loss)
    loss_dict_dev.append(dev_loss)
    end_time = time.time()
    epoch_min, epoch_sec = record_time(start_time, end_time)
    logger.info(
        f'epoch: [{epoch:02}/{max_epoch}]  train_loss={train_loss:.3f}  val_loss={dev_loss:.3f}  '
        f'ppl={dev_ppl:.2f}  time: {epoch_min}m {epoch_sec}s')

if max_epoch > 0:
    save_checkpoint(experiment_time=cur_time, model=model, optimizer=optimizer)

show_plot(loss_dict_train, loss_dict_dev)

bleu_scorer = BleuScore()
bleu_scorer.reset_gens()
bleu_scorer.set_refs(data_processor.get_refs())

for idx_data in range(len(data_processor.test_data)):
    seq_input = torch.tensor(data_processor.process_one_data(idx_data=idx_data),
                             dtype=torch.float, device=device).unsqueeze(dim=0)
    seq_output = evaluate(seq_input)
    list_seq = seq_output.squeeze().tolist()
    text_gen = data_processor.translate(list_seq)
    bleu_scorer.add_gen(text_gen)

bleu_score = bleu_scorer.calculate()
logger.info(f'bleu score: {bleu_score:.4f}')
