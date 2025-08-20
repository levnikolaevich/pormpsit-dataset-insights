import sys
import argparse
import traceback
import logging
import json
import yaml


def initialization():
    parser = argparse.ArgumentParser()
    parser.add_argument('dlcounts', type=argparse.FileType('r'), help="Input domain labels counts file")
    parser.add_argument('yamlfile', type=argparse.FileType('a'), help="Output YAML stats file.")
    return parser.parse_args()


def main():
    args = initialization()
    stats = {}
    dl_dict = {}
    for line in args.dlcounts:
        parts = line.split()
        label = parts[1].strip()
        count = int(parts[0].strip())
        dl_dict[label] = count
    if dl_dict:
        stats["domain_labels"] = json.dumps(dl_dict)
    yaml.dump(stats, args.yamlfile)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        logging.error(tb)
        sys.exit(1)
