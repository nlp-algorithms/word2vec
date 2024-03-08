import argparse
from preprocess.preprocess import Preprocesser
from generate_embeddings.generate_embeddings import GenerateEmbeddings
import logging
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--target", help="[preprocess, generate_embeddings]",choices=["preprocess","generate_embeddings"],required=True)
parser.add_argument("--context_window", help="[preprocess, generate_embeddings]",required=False, default=2)
parser.add_argument("--corpus", help="dir of text files",required=False, default="data/test_corpus/")
parser.add_argument("--data_path", help="dir of text files",required=False, default="vocab.pickle")

args = parser.parse_args()

if(args.target == 'preprocess'):
    Preprocesser(context_window=args.context_window, directory=args.corpus).process()
if(args.target == 'generate_embeddings'):
    GenerateEmbeddings(data_path=args.data_path,context_window=args.context_window).train()