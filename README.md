# COMP4560 project 

This project contains two stages. The first stage is image captioning while the 
second stage is knowledge retrieval in StackOverFlow dataset.  

The `.py` files in `image_caption` package and `knowledge_retrieval` are code for stage 1 and 
stage 2 respectively.

Inside `image_caption` package, run `caption_training.py` to train and test the image 
captioning model. We used the existing python package **nlg-eval** for predicted caption evaluation. 
(https://github.com/Maluuba/nlg-eval)

Inside `knowledge_retrieval` package, run `stackoverflow_searching.py` to provide top 5 relevant
posts given the selected 50 queries and run `stackoverflow_evaluation.py` to get top 5 accuracy 
and MRR score.

