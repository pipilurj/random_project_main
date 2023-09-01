_base_ = ['../_base_/dataset/mix_pretrain_concat3.py', '../_base_/model/shikra.py', '../_base_/train/shikra_deepspeed_lora.py']

data_args = dict(
    #
    train=dict(
        _delete_=True,
        type='ConcatDataset',
        cfgs=[
            {{_base_.DEFAULT_TRAIN_DATASET.rec}},
            # {{_base_.DEFAULT_TRAIN_DATASET.recvg}},
        ],
    )
)
training_args = dict(
    save_steps=5000,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    lora_enable=False,
    output_dir='./exp/shikra3/{{fileBasenameNoExtension}}',
)

model_args = dict(
    type="shikra_continuous",
    conv_args=dict(
        tokenize_kwargs=dict(truncation_size=4096),
    ),
    model_name_or_path="/home/pirenjie/pretrained_weights/llava_7b",
    target_processor=dict(
        boxes=dict(type='TokenFormatterContinuous', num_bins=16),
    ),
    process_func_args=dict(
        target=dict(type='BoxFormatProcessContinous'),
    ),
)
