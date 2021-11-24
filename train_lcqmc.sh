dataset="LCQMC"
model_type="bert-base-chinese"
batch_size=32
epochs=5
seed=2048
learning_rate='3e-5'
train_file="data/$dataset/clean/train_clean.txt"
dev_file="data/$dataset/clean/dev_clean.txt"
test_file="data/$dataset/clean/test_clean.txt"
echo $train_file
CUDA_VISIBLE_DEVICES=7 python3 Train_baseline.py \
--train_file $train_file \
--dev_file $dev_file \
--test_file $test_file \
--save_dir "result/$dataset/bs$batch_size/epoch$epochs/seed$seed/$learning_rate/checkpoints" \
--do_train True \
--do_lower_case True \
--seed $seed \
--learning_rate $learning_rate \
--epochs $epochs \
--batch_size $batch_size \
--max_length 128 \
--warmup_steps 0.1 \
#--mlm \
#--mlm_weight $mlm_weight
