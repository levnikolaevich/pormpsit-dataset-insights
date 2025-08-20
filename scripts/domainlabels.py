import sys
import os
import io
import argparse
import logging
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from util import logging_setup


def initialization():
    parser = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=argparse.FileType('rt', errors="replace"),
        default=io.TextIOWrapper(sys.stdin.buffer, errors="replace"),
        help="Input sentences.",
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=argparse.FileType('wt'),
        default=sys.stdout,
        help="Output of the domain identification.",
    )

    groupO = parser.add_argument_group("Options")
    groupO.add_argument(
        "--field",
        type=str,
        default="text",
        help="Name of the JSON field that contains the text to be analyzed",
    )
    groupO.add_argument("--raw", action="store_true", help="True if the input is already raw, non-json text")
    groupO.add_argument("--batchsize", type=int, default=256, help="GPU batch size")

    groupL = parser.add_argument_group("Logging")
    groupL.add_argument('-q', '--quiet', action='store_true', help='Silent logging mode')
    groupL.add_argument('--debug', action='store_true', help='Debug logging mode')
    groupL.add_argument('--info', action='store_true', help='Info logging mode')
    groupL.add_argument('--logfile', type=argparse.FileType('a'), default=sys.stderr, help="Store log to a file")

    args = parser.parse_args()
    logging_setup(args)
    return args


class DomainLabels:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = "EuropeanParliament/eurovoc_2025"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id).to(self.device)
        logging.info("Model loaded")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        logging.info("Tokenizer loaded")

    def get_labels_batch(self, docs_text):
        # Filter out empty or None texts to avoid tokenizer/model errors
        filtered_texts = [t for t in docs_text if t]
        if not filtered_texts:
            return []

        inputs = self.tokenizer(
            filtered_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.model(**inputs).logits
            else:
                # On CPU, avoid float16 autocast; optionally could use bfloat16 if available
                logits = self.model(**inputs).logits
        predicted = torch.argmax(logits, dim=1).cpu().tolist()
        id2label = self.model.config.id2label
        return [id2label[i] for i in predicted]


def perform_identification(args):
    dl = DomainLabels()
    buffer = []
    for line in args.input:
        if not args.raw:
            doc = json.loads(line)
            doc_text = doc.get(args.field)
        else:
            doc_text = line
        if doc_text:
            buffer.append(doc_text)
        if len(buffer) < args.batchsize:
            continue
        labels = dl.get_labels_batch(buffer)
        buffer = []
        for l in labels:
            args.output.write(l.strip() + "\n")
    if buffer:
        labels = dl.get_labels_batch(buffer)
        for l in labels:
            args.output.write(l.strip() + "\n")


def main():
    args = initialization()
    perform_identification(args)


if __name__ == "__main__":
    main()
