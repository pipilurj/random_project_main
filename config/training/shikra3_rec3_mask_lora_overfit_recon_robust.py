_base_ = ['../_base_/dataset/DEFAULT_TRAIN_DATASET.py', '../_base_/dataset/DEFAULT_TEST_REC_VARIANT.py',
          '../_base_/model/shikra.py', '../_base_/train/shikra_deepspeed_lora.py']
data_args = dict(
    #
    # _delete_=True,
    train=dict(
        type='ConcatDataset',
        cfgs=[
            # {{_base_.DEFAULT_TRAIN_DATASET.rec}},
            {{_base_.DEFAULT_TRAIN_DATASET.rec_mask_subset}}
            # {{_base_.DEFAULT_TRAIN_DATASET.recvg}},
        ],
    ),
    validation=None,
    test=None,

    # compute_metric
    compute_metric=None,

    # padding collator kwargs
    collator_kwargs=dict(
        padding=True,
        max_length=1024,
    ),

    # generate config
    gen_kwargs=dict(
        max_new_tokens=1024,
        num_beams=1,
    ),
)
training_args = dict(
    save_steps=5000,
    num_train_epochs=1000,
    per_device_train_batch_size=8,
    lora_enable=False,
    output_dir='./exp/shikra3/{{fileBasenameNoExtension}}',
)

model_args = dict(
    type="shikra_mask",
    conv_args=dict(
        tokenize_kwargs=dict(truncation_size=4096),
    ),
    model_name_or_path="/home/pirenjie/pretrained_weights/llava_7b",
    target_processor=dict(
        boxes=dict(type='BoxMaskFormatter'),
    ),
    lora_enable=True,
    lora_r = 32,
    lora_alpha = 32,
    lora_dropout = 0.1,
    pretrained_autoencoder="/home/pirenjie/transformer-master/saved/pixelmask_resnet_dist0.01_noise0.01_triplet1-model.pt",
    freeze_autoencoder=True,
)
