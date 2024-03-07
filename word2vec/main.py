import argparse
from preprocess.preprocess import Preprocesser
import logging
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--target", help="[preprocess, generate_embeddings]",choices=["preprocess","generate_embeddings"],required=True)
parser.add_argument("--context_window", help="[preprocess, generate_embeddings]",required=False, default=2)
parser.add_argument("--corpus", help="dir of text files",required=False, default="data/test_corpus")
args = parser.parse_args()

if(args.target == 'preprocess'):
    Preprocesser(context_window=args.context_window, directory=args.corpus)