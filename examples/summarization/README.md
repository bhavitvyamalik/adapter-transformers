# Text Summarization with Pretrained Encoders

This folder contains part of the code necessary to reproduce the results on abstractive summarization from the article [Text Summarization with Pretrained Encoders](https://arxiv.org/pdf/1908.08345.pdf) by [Yang Liu](https://nlp-yang.github.io/) and [Mirella Lapata](https://homepages.inf.ed.ac.uk/mlap/). It can also be used to summarize any document.

The original code can be found on the Yang Liu's [github repository](https://github.com/nlpyang/PreSumm).

The model is loaded with the pre-trained weights for the abstractive summarization model trained on the CNN/Daily Mail dataset with an extractive and then abstractive tasks.

## Setup

```
git clone https://github.com/huggingface/transformers && cd transformers
pip install [--editable] .
pip install nltk py-rouge
cd examples/summarization
```

## Reproduce the authors' results on ROUGE

To be able to reproduce the authors' results on the CNN/Daily Mail dataset you first need to download both CNN and Daily Mail datasets [from Kyunghyun Cho's website](https://cs.nyu.edu/~kcho/DMQA/) (the links next to "Stories") in the same folder. Then uncompress the archives by running:

```bash
tar -xvf cnn_stories.tgz && tar -xvf dailymail_stories.tgz
```

And move all the stories to the same folder. We will refer as `$DATA_PATH` the path to where you uncompressed both archive. Then run the following in the same folder as `run_summarization.py`:

```bash
python run_summarization.py \
    --documents_dir $DATA_PATH \
    --summaries_output_dir $SUMMARIES_PATH \ # optional
    --visible_gpus 0,1,2 \
    --batch_size 4 \
    --min_length 50 \
    --max_length 200 \
    --beam_size 5 \
    --alpha 0.95 \
    --block_trigram true \
    --compute_rouge true
```

The ROUGE scores will be displayed in the console at the end of evaluation and written in a `rouge_scores.txt` file.

## Summarize any text

Put the documents that you would like to summarize in a folder (the path to which is referred to as `$DATA_PATH` below) and run the following in the same folder as `run_summarization.py`:

```bash
python run_summarization.py \
    --documents_dir $DATA_PATH \
    --summaries_output_dir $SUMMARIES_PATH \ # optional
    --visible_gpus 0,1,2 \
    --batch_size 4 \
    --min_length 50 \
    --max_length 200 \
    --beam_size 5 \
    --alpha 0.95 \
    --block_trigram true \
```

If you want to compute ROUGE on another dataset you will need to tweak the stories/summaries import in `utils_summarization.py`