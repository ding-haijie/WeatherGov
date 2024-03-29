import json
import traceback
import urllib.parse as url_parser
from http.server import HTTPServer, BaseHTTPRequestHandler

import torch

from config import params
from data_preprocess.get_info import get_info
from data_processor import DataProcessor
from models import EncoderGate, DecoderAttn, Data2Text
from utils import load_checkpoint, get_logger


class HttpHandler(BaseHTTPRequestHandler):
    def _response(self, path, args):
        status_code = 200
        resp = {'status_code': 0, 'msg': '', 'value': ''}
        try:
            if args is not None:
                args = json.loads(args)
            else:
                args = {}

            if path == '/':
                resp['value'] = 'root dir'
            elif path == '/weather':
                events = args.get("events")
                seq_info_numpy = get_info(self.parse_event(events))
                seq_input = torch.tensor(seq_info_numpy, dtype=torch.float,
                                         device=device).unsqueeze(dim=0).permute(1, 0, 2)
                resp['value'] = str(self.generate_text(seq_input))
            else:
                status_code = 404
                resp['status_code'] = 404
                resp['msg'] = "path: [" + path + "] does not exist !"

        except Exception as e:
            resp['status_code'] = 1
            resp['msg'] = "server error: " + str(e) + "\n" + traceback.format_exc()

        try:
            resp = json.dumps(resp, ensure_ascii=False)
        except Exception as e:
            resp = {'status_code': 2, 'msg': 'server error: ' +
                                             str(e) + "\n" + traceback.format_exc(), 'value': ""}
            resp = json.dumps(resp, ensure_ascii=False)

        self.send_response(status_code)
        self.send_header('Content-type', 'text/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(resp.encode())

        logger.info(
            f'{self.client_address[0]} -- [{self.log_date_time_string()}] "{self.requestline}" {status_code}')

    @staticmethod
    def generate_text(gen_input):
        sos_token = torch.tensor([[0]], dtype=torch.long, device=device)
        with torch.no_grad():
            seq_output, attn = model(gen_input, sos_token, train_mode=False)
        text_gen = data_processor.translate(seq_output.squeeze().tolist())

        return {"text": text_gen, "attn": str(attn.squeeze().numpy().tolist())}

    # noinspection DuplicatedCode
    @staticmethod
    def parse_event(events):
        _dict = {}
        idx = 0
        for line in events.split("\n"):
            if len(line.strip()) == 0:
                continue
            dict_idx = {}
            for text in line.split():
                if '.type' in text:
                    dict_idx['type'] = text[text.index(':') + 1:]
                if '.label' in text:
                    dict_idx['label'] = text[text.index(':') + 1:]
                if '@time' in text:
                    dict_idx['time'] = text[text.index(':') + 1:]
                if '#min' in text:
                    dict_idx['min'] = text[text.index(':') + 1:]
                if '#mean' in text:
                    dict_idx['mean'] = text[text.index(':') + 1:]
                if '#max' in text:
                    dict_idx['max'] = text[text.index(':') + 1:]
                if '@mode' in text:
                    dict_idx['mode'] = text[text.index(':') + 1:]
                if '@mode-bucket-0-20-2' in text:
                    dict_idx['mode_bucket_0_20_2'] = text[text.index(':') + 1:]
                if '@mode-bucket-0-100-4' in text:
                    dict_idx['mode_bucket_0_100_4'] = text[text.index(':') + 1:]
            if 'mode_bucket_0_20_2' not in dict_idx:
                dict_idx['mode_bucket_0_20_2'] = ''
            _dict['id' + str(idx)] = dict_idx
            idx += 1
        return _dict

    def do_GET(self):
        result = url_parser.urlparse(self.path)
        self._response(result.path, result.query)

    def do_POST(self):
        path = url_parser.urlparse(self.path).path
        content_len = int(self.headers.get('Content-Length'))
        post_body = self.rfile.read(content_len)
        self._response(path, post_body.decode())


if __name__ == '__main__':
    logger = get_logger('./results/logs/server.log')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_processor = DataProcessor()
    encoder = EncoderGate(params["input_dim"], params["embed_dim"], params["hidden_dim"], params["dropout"])
    decoder = DecoderAttn(params["output_dim"], params["embed_dim"], params["hidden_dim"], params["dropout"])
    model = Data2Text(encoder, decoder, params["beam_width"], torch.cuda.is_available()).to(device)
    checkpoint = load_checkpoint(latest=True, device=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    del checkpoint

    httpd = HTTPServer(('', 9527), HttpHandler)
    httpd.serve_forever()
