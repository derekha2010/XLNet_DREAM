## XLNet baseline for DREAM dataset 
Author: Chenglei Si (River Valley High School, Singapore)

Update:
Sometimes you may get degenerate runs where the performance is far lower than the expected performance. This is mainly because the training is not stable on smaller datasets. You may try to change the random seeds (and perhaps learning rate, batch size, warmup steps or other hyperparameters as well) and restart training. If you want, I can send you a trained checkpoint. Feel free to contact me through email: sichenglei1125@gmail.com  
Note:
You should use the dev set to do hyper-parameter tuning and then use the test file and trained model to evaluate on the test data. This is the standard practice for ML. 

Usage:

1. Download data and unzip to this folder.
2. (If you have not installed sentencepiece) Run `pip install sentencepiece`
3. Run `sh run.sh`
4. To test a trained model, Run `python test_xlnet_dream.py --data_dir=data --xlnet_model=xlnet-large-cased --output_dir=xlnet_dream --checkpoint_name=pytorch_model_3epoch_72_len256.bin --max_seq_length=256 --do_eval --eval_batch_size=1` You may need to change the checkpint name accordingly. 

(The hyperparameters that I used can be found in run.sh)

Result: 72.0 (SOTA as of July 2019, [leaderboard](https://dataset.org/dream/))

Note: My codes are built upon huggingface's implementation of [pytorch_transformers](https://github.com/huggingface/pytorch-transformers), and the original XLNet paper is: [(Yang et al., 2019)](https://arxiv.org/pdf/1906.08237.pdf).

More build info:
Pull this:
git clone https://github.com/derekha2010/XLNet_DREAM

Dream datasets:
git clone https://github.com/nlpdata/dream
cp dream/data XLNet_DREAM

Install apex in windows:
git clone https://github.com/kezewang/apex
cd apex
modify setup.py (104) to version_dependent_macros = version_ge_1_1 + version_ge_1_3 + version_ge_1_5 + ['-D_ENABLE_EXTENDED_ALIGNED_STORAGE']
set USE_NINJA=OFF
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

Other package:
pip install sentencepiece
