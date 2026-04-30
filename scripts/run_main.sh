device=0
modality=MF
lr=1e-3
wd=0
num_experts=12
num_layers_fus=1
top_k=4
train_epochs=175
warm_up_epochs=5
hidden_dim=128
num_patches=16
batch_size=8
num_heads=4
gate_loss_weight=1e-2
align_loss_weight=1e-2
crossmod_loss_weight=1e-2

CUDA_VISIBLE_DEVICES=$device python main.py \
    --data_dir /path/to/data \
    --n_runs 5 \
    --runparallel False \
    --run 0 \
    --modality $modality \
    --lr $lr \
    --wd $wd \
    --num_experts $num_experts \
    --num_layers_fus $num_layers_fus \
    --top_k $top_k \
    --train_epochs $train_epochs \
    --warm_up_epochs $warm_up_epochs \
    --hidden_dim $hidden_dim \
    --num_patches $num_patches \
    --batch_size $batch_size \
    --num_heads $num_heads \
    --gate_loss_weight $gate_loss_weight \
    --align_loss_weight $align_loss_weight \
    --crossmod_loss_weight $crossmod_loss_weight \
    --save True \
#    --load_model True \

