_base_ = ['_base_/dataset/mix_pretrain_concat3.py', '_base_/model/shikra.py', '_base_/train/shikra_deepspeed_lora.py']

training_args = dict(
    num_train_epochs=1,
    output_dir='./exp/{{fileBasenameNoExtension}}',
)

model_args = dict(
    conv_args=dict(
        tokenize_kwargs=dict(truncation_size=4096),
    ),
    model_name_or_path="/home/pirenjie/pretrained_weights/llava_7b",
)
