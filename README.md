## Code Structure

### 1. trains/
- code used for fine-tuning CLIP with our augmented COCO dataset
- change the code in dataset.py to load images and captions for training, run fine_tune.py to start the training process

### 2.openclip/
- code used to load CLIP models adopted from official CLIP repository [CLIP](https://github.com/openai/CLIP)


### 3.evaluation/
- code for evaluation on SAN-Bench
- modify the model path and run the python files to get evaluation resluts
### 4.preparation/
- code to pre-process Flickr30k for constructing SAN-Bench


## Image dataset
1. Flickr30k
   - this is used for SAN-Bench matching task
   - download images from [flickr30k](https://huggingface.co/datasets/nlphuji/flickr30k)
   - change the image path when load ''SAN-Bench_matching_task_captions.csv''

2. SAN-Bench Sythetic Images
   - this is used for SAN-Bench retrieval task
   - We will upload our sythetic images in near feature

3. COCO_2017 
   - this is used for fine-tuning CLIP to build SAN-CLIP
   - download images from [COCO_2017](https://cocodataset.org/#download)
   - change the image path when load ''coco2017_fine-tuning.csv''

