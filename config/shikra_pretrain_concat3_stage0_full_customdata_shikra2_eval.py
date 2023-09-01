_base_ = ['_base_/dataset/DEFAULT_TEST_REC_VARIANT.py', '_base_/model/shikra.py', '_base_/train/eval.py']

data_args = dict(
    train=None,
    # validation={k: {'cfg': v, 'compute_metric': dict(type='RECComputeMetrics')} for k, v in _base_.DEFAULT_TEST_REC_VARIANT.items()},
    # validation=_base_.DEFAULT_TEST_FLICKR_VARIANT.FLICKR_EVAL_with_box_subset,
    validation=_base_.DEFAULT_TEST_REC_VARIANT.REC_REFCOCOG_UMD_TEST,
    # validation=_base_.DEFAULT_TEST_REC_VARIANT.REC_REFCOCOG_UMD_TEST_subset,
    test=None,
    multitest=None,

    compute_metric=dict(type='RECComputeMetrics2'),

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
    do_train=False,
    do_eval=True,
    do_predict=False,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    lora_enable=False,
    output_dir='./exp/{{fileBasenameNoExtension}}',
)

model_args = dict(
    type="shikra_detr",
    conv_args=dict(
        tokenize_kwargs=dict(truncation_size=4096),
    ),
    model_name_or_path="/home/pirenjie/pretrained_weights/llava_7b",
    target_processor=dict(
        boxes=dict(type='TokenFormatter2', num_bins=16),
        # boxes=dict(type='RetFormatter'),
    ),
)
