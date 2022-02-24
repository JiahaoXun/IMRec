# IMRec

****

The dataset for paper "Why Do We Click: Visual Impression-aware News Recommendation", ACM MM 2021

****

**Dataset**

- The filename corresponds to the newsid in the original dataset.
- The download link for our dataset: https://drive.google.com/file/d/1gx0OzN7qSuyRlvN0cfVUjB4tmoKvvQk1/view?usp=sharing
- imageList.npy: The newsid list of successfully crawled images in the format of [newsid,...].

**Utils**

- data_generator.py：For generating IM-MIND dataset (news  text and cover images are required).
- word_feature_generator.py：For generating visual impression representations of words.
- global_feature_generator.py：For generating global impression representations.
