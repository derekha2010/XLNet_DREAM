python run_xlnet_dream.py --data_dir=data --xlnet_model=xlnet-large-cased --output_dir=xlnet_dream --max_seq_length=256 --do_train --do_eval --train_batch_size=32 --eval_batch_size=1 --learning_rate=1e-5 --num_train_epochs=4 --warmup_proportion=0.1 --gradient_accumulation_steps=32 && /root/shutdown.sh
python run_xlnet_dream.py --data_dir=data --xlnet_model=xlnet-large-cased --output_dir=xlnet_dream --max_seq_length=128 --do_train --do_eval --train_batch_size=32 --eval_batch_size=2 --learning_rate=2e-5 --num_train_epochs=3 --warmup_steps=120 --weight_decay=0.0 --adam_epsilon=1e-8 --gradient_accumulation_steps=16 && /root/shutdown.sh
