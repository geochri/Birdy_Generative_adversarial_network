#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 10:09:34 2019

@author: tsd
"""

from src.crawl_googleimages import GoogleImageExtractor
from src.image_generate import train_and_fit
from src.misc import save_as_pickle
from argparse import ArgumentParser
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--crawl", type=int, default=0, help="0: Don't crawl for new images, 1: Crawl it")
    parser.add_argument("--train_data", type=str, default="./data/_01_Jan20/", help="Training data folder containing images")
    parser.add_argument("--model_no", type=int, default=1, help="0: vanilla GAN, 1: DCGAN")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=750, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    parser.add_argument("--gradient_acc_steps", type=int, default=1, help="Number of steps of gradient accumulation")
    args = parser.parse_args()
    save_as_pickle("args.pkl", args)
    
    if args.crawl == 1:
        w = GoogleImageExtractor('')
        searchlist_filename = "./data/search_terms.txt"
        w.set_num_image_to_dl(1000)
        w.get_searchlist_fr_file(searchlist_filename) #replace the searclist
        w.multi_search_download()
        
    output = train_and_fit(args)