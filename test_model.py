import time
import datetime

from models import EncoderGate, DecoderAttn, Data2Text
from data_processor import DataProcessor
from utils import *
from config import params

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_text(gen_input, net):
    net.eval()
    sos_token = torch.tensor([[0]], dtype=torch.long, device=device)
    with torch.no_grad():  # truncate back_propagation
        seq_output, _ = net(gen_input, sos_token, train_mode=False)
    text_gen = data_processor.translate(seq_output.squeeze().tolist())

    return text_gen


if __name__ == '__main__':

    cur_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M')
    input_dim = params["input_dim"]
    output_dim = params["output_dim"]
    embed_dim = params["embed_dim"]
    hidden_dim = params["hidden_dim"]
    dropout = params["dropout"]
    beam_width = params["beam_width"]

    data_processor = DataProcessor()
    print('finish processing data !')

    encoder = EncoderGate(input_dim, embed_dim, hidden_dim, dropout)
    decoder = DecoderAttn(output_dim, embed_dim, hidden_dim, dropout)
    model = Data2Text(encoder, decoder, beam_width).to(device)

    # load stat_dict
    checkpoint = load_checkpoint(latest=True)
    model.load_state_dict(checkpoint['model'])
    print('finish loading model !')

    # generates 10 examples for preview
    rand_list = random_list(0, len(data_processor.test_data), 10)
    print(f'random_list: {rand_list}')
    check_file_exist('./results/utterances')
    start = time.time()

    with open('./results/utterances/' + cur_time + '.txt', 'w') as f:
        f.write(cur_time + '\n')
        for idx_data in rand_list:
            seq_input = torch.tensor(data_processor.process_one_data(
                idx_data=idx_data), dtype=torch.float, device=device).unsqueeze(dim=0).permute(1, 0, 2)
            text = generate_text(seq_input, model)
            f.write('                    system \n')
            f.write(text + '\n')
            f.write('                     gold \n')
            f.write(data_processor.test_data[idx_data]['text'] + '\n')
            f.write('################################################## \n\n')

    duration = time.time() - start
    print(f'average duration: {(duration / 10):.2f}')
