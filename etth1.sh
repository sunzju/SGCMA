export CUDA_VISIBLE_DEVICES=2

seq_len=336

for pred_len in 96 192 336 720
do

python run_exp.py \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --patience 5

echo '====================================================================================================================='
done
