_base_ = ['_base_/dataset/mix_pretrain_concat3.py', '_base_/model/shikra.py', '_base_/train/shikra_deepspeed_lora.py']

data_args = dict(
    #
    train=dict(
        _delete_=True,
        type='ConcatDataset',
        cfgs=[
            {{_base_.DEFAULT_TRAIN_DATASET.rec}},
            # {{_base_.DEFAULT_TRAIN_DATASET.recvg}},
            # {{_base_.DEFAULT_TRAIN_DATASET.llavacc3m}},
            # {{_base_.DEFAULT_TRAIN_DATASET.llavalcs}},
            # {{_base_.DEFAULT_TRAIN_DATASET.instruct}},
            # {{_base_.DEFAULT_TRAIN_DATASET.caption}},
            # {{_base_.DEFAULT_TRAIN_DATASET.gc}},
            # {{_base_.DEFAULT_TRAIN_DATASET.flickr}},
        ],
    )
)
training_args = dict(
    num_train_epochs=1,
    per_device_train_batch_size=8,
    lora_enable=False,
    output_dir='./exp/{{fileBasenameNoExtension}}',
)

model_args = dict(
    conv_args=dict(
        tokenize_kwargs=dict(truncation_size=4096),
    ),
    model_name_or_path="/home/pirenjie/pretrained_weights/llava_7b",
)
