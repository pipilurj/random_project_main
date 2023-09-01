_base_ = ['_base_/dataset/mix_pretrain_concat3.py', '_base_/model/shikra.py', '_base_/train/shikra_deepspeed_lora.py']

data_args = dict(
    #
    train=dict(
        _delete_=True,
        type='ConcatDataset',
        cfgs=[
            {{_base_.DEFAULT_TRAIN_DATASET.rec_subset}},
            {{_base_.DEFAULT_TRAIN_DATASET.recvg_subset}},
            {{_base_.DEFAULT_TRAIN_DATASET.llavacc3m}},
            {{_base_.DEFAULT_TRAIN_DATASET.instruct}},
        ],
    )
)
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
