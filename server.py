import json
import traceback
import urllib.parse as url_parser
from http.server import HTTPServer, BaseHTTPRequestHandler

import mysql.connector as mysql_connector
import torch

from config import params
from data_preprocess.get_info import get_info
from data_processor import DataProcessor
from models import EncoderGate, DecoderAttn, Data2Text
from utils import load_checkpoint


# noinspection DuplicatedCode
class HttpHandler(BaseHTTPRequestHandler):
    def _response(self, path, args):
        status_code = 200
        resp = {'status_code': 0, 'msg': '', 'value': ''}
        try:
            if args is not None:
                args = url_parser.parse_qs(args).items()
                args = dict([(k, v[0]) for k, v in args])
            else:
                args = {}

            if path == '/':
                resp['value'] = 'root directory'
            elif path == "/weather":
                date, offset = args.get("date"), args.get("offset")
                state, city = args.get("state"), args.get("city")
                cursor.execute(
                    "select events from d2t_rel.weather where date=%s and "
                    "offset=%s and state=%s and city=%s", (date, offset, state, city))
                events = cursor.fetchall()[0][0]
                seq_info_numpy = get_info(self.parse_event(events))
                seq_input = torch.tensor(seq_info_numpy, dtype=torch.float,
                                         device=device).unsqueeze(dim=0).permute(1, 0, 2)
                resp['value'] = str(self.generate_text(seq_input))
            else:
                status_code = 404
                resp["status_code"] = 404
                resp["msg"] = 'path: [' + path + "] does not exist !"

        except Exception as e:
            resp["status_code"] = 1
            resp["msg"] = 'server error: ' + str(e) + "\n" + traceback.format_exc()

        try:
            resp = json.dumps(resp, ensure_ascii=False)
        except Exception as e:
            resp = {'status_code': 2, 'msg': 'server error: ' +
                                             str(e) + "\n" + traceback.format_exc(), 'value': ''}
            resp = json.dumps(resp, ensure_ascii=False)

        self.send_response(status_code)
        self.send_header('Content-type', 'text/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(resp.encode())

    @staticmethod
    def generate_text(gen_input):
        sos_token = torch.tensor([[0]], dtype=torch.long, device=device)
        with torch.no_grad():
            seq_output, attn = model(gen_input, sos_token, train_mode=False)
        text_gen = data_processor.translate(seq_output.squeeze().tolist())

        return {"text": text_gen, "attn": str(attn.squeeze().numpy().tolist())}

    @staticmethod
    def parse_event(events):
        data_dict = {}
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
            data_dict['id' + str(idx)] = dict_idx
            idx += 1
        return data_dict

    def do_GET(self):
        result = url_parser.urlparse(self.path)
        self._response(result.path, result.query)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_processor = DataProcessor()

    encoder = EncoderGate(params["input_dim"], params["embed_dim"], params["hidden_dim"], params["dropout"])
    decoder = DecoderAttn(params["output_dim"], params["embed_dim"], params["hidden_dim"], params["dropout"])
    model = Data2Text(encoder, decoder, params["beam_width"], torch.cuda.is_available()).to(device)
    checkpoint = load_checkpoint(latest=True, device=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    del checkpoint

    conn = mysql_connector.connect(host="[host]", port="[port]",
                                   user="[username]", password="[password]",
                                   database="[db_name]")
    cursor = conn.cursor()

    httpd = HTTPServer(('', 9527), HttpHandler)
    httpd.serve_forever()
