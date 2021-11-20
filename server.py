import json
import traceback
import urllib.parse as url_parser
from http.server import HTTPServer, BaseHTTPRequestHandler

from config import params
from data_processor import DataProcessor
from models import EncoderGate, DecoderAttn, Data2Text
from utils import *


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
                idx_data = args.get('id', '')
                seq_input = torch.tensor(data_processor.process_one_data(
                    idx_data=int(idx_data)), dtype=torch.float,
                    device=device).unsqueeze(dim=0).permute(1, 0, 2)
                resp['value'] = self.generate_text(seq_input)
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
        with torch.no_grad():  # truncate back_propagation
            seq_output, attn = model(gen_input, sos_token, train_mode=False)
        text_gen = data_processor.translate(seq_output.squeeze().tolist())

        return {'text': text_gen, 'attn': attn.numpy().tolist()}

    def do_GET(self):
        result = url_parser.urlparse(self.path)
        self._response(result.path, result.query)

    def do_POST(self):
        args = self.rfile.read(
            int(self.headers['content-length'])).decode("utf-8")
        self._response(self.path, args)


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

    httpd = HTTPServer(('127.0.0.1', 9527), HttpHandler)
    httpd.serve_forever()
