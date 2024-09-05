In this [experiment](Finetuning-Whisper-ASR-task.ipynb), I fine-tuned OpenAI's Whisper model for automatic speech recognition on the Dhivehi language subset of the Common Voice 13 dataset using Hugging Face's `transformers` library.  (I followed the data preprocessing and training steps in this [tutorial](https://huggingface.co/learn/audio-course/chapter5/fine-tuning) in the Hugging Face Audio course, but wrote my own scripts.)

The goal was to transcribe spoken Dhivehi language, using the Sinhalese version of `openai/whisper-small`, a pre-trained language model for automatic speech recognition. Sinhalese was chosen because it shares linguistic similarities with Dhivehi.

## Results

Before fine-tuning, the pre-trained Whisper model had an orthographic Word Error Rate (WER) of 167% and normalized WER of 126% on the Dhivehi test set, indicating poor performance. After fine-tuning, the orthographic and normalizes WERs were drastically reduced to 62% and 13%. 

| Step | Training Loss | Validation Loss | Wer Ortho | Wer Normalized |
|------|---------------|-----------------|-----------|----------------|
| 250  | 0.196500      | 0.225315        | 72.149871 | 16.613346       |
| 500  | 0.117400      | 0.170857        | 62.462567 | 12.781236       |


I learned the process of instantiating the necessary components for the ASR pipeline, including the feature extractor, tokenizer, and processor. I experimented with the ASR pipeline, from loading and preprocessing the dataset to defining custom evaluation metrics and optimizing training with gradient checkpointing and mixed precision.

This enhancement underscores the effectiveness of the fine-tuning process on previously unsupported languages. 