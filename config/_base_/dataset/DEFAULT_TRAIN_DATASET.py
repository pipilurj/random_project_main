_base_ = [
    'DEFAULT_TRAIN_GQA_VARIANT.py',
    'DEFAULT_TRAIN_CLEVR_VARIANT.py',
    'DEFAULT_TRAIN_POINT_VARIANT.py',
    'DEFAULT_TRAIN_GPTGEN_VARIANT.py',
    'DEFAULT_TRAIN_VCR_VARIANT.py',
    'DEFAULT_TRAIN_VQAv2_VARIANT.py',
    'DEFAULT_TRAIN_VQAEX_VARIANT.py',
]

DEFAULT_TRAIN_DATASET = dict(
    flickr=dict(
        type='FlickrDataset',
        filename=r'{{fileDirname}}/../../../data/CWB_flickr30k_train.jsonl',
        image_folder=r'/home/pirenjie/data/flickr30k/flickr30k-images',
        template_file=r'{{fileDirname}}/template/flickr30k.json',
    ),
    rec=dict(
        type='RECDataset',
        filename=r'{{fileDirname}}/../../../data/REC_ref3_train.jsonl',
        image_folder=r'/home/pirenjie/data/refcoco/images/train2014',
        template_file=r'{{fileDirname}}/template/REC.json',
    ),
    rec_simple=dict(
        type='RECDataset',
        filename=r'{{fileDirname}}/../../../data/REC_ref3_train.jsonl',
        image_folder=r'/home/pirenjie/data/refcoco/images/train2014',
        template_file=r'{{fileDirname}}/template/REC_img_exp.json',
    ),
    rec_simple_subset=dict(
        type='RECRETDataset',
        filename=r'{{fileDirname}}/../../../data/REC_ref3_train_subset10.jsonl',
        image_folder=r'/home/pirenjie/data/refcoco/images/train2014',
        template_file=r'{{fileDirname}}/template/REC_img_exp.json',
    ),
    rec_subset=dict(
        type='RECDataset',
        filename=r'{{fileDirname}}/../../../data/REC_ref3_train_subset100.jsonl',
        image_folder=r'/home/pirenjie/data/refcoco/images/train2014',
        template_file=r'{{fileDirname}}/template/REC.json',
    ),
    rec_mask_all=dict(
        type='REFMaskDataset',
        mask_dir="/home/pirenjie/data/refcoco/shikra_mask/masks",
        filename=r'/home/pirenjie/data/refcoco/shikra_mask/anns/train_mask_merged.jsonl',
        image_folder=r'/home/pirenjie/data/refcoco/images/train2014',
        template_file=r'{{fileDirname}}/template/REC_img_exp.json',
    ),
    rec_mask=dict(
        type='REFMaskDataset',
        mask_dir="/home/pirenjie/data/refcoco/shikra_mask/masks",
        filename=r'/home/pirenjie/data/refcoco/shikra_mask/anns/refcocog/train_mask.jsonl',
        image_folder=r'/home/pirenjie/data/refcoco/images/train2014',
        template_file=r'{{fileDirname}}/template/REC_img_exp.json',
    ),
    rec_mask_subset=dict(
        type='REFMaskDataset',
        mask_dir="/home/pirenjie/data/refcoco/shikra_mask/masks",
        filename=r'/home/pirenjie/data/refcoco/shikra_mask/anns/refcocog/train_mask_subset.jsonl',
        image_folder=r'/home/pirenjie/data/refcoco/images/train2014',
        template_file=r'{{fileDirname}}/template/REC_img_exp.json',
    ),
    recvg=dict(
        type='RECDataset',
        filename=r'{{fileDirname}}/../../../data/GC_genome196_train.jsonl',
        image_folder=r'/home/pirenjie/data/vg_1.2',
        template_file=r'{{fileDirname}}/template/REC.json',
    ),
    recvg_subset500k=dict(
        type='RECDataset',
        filename=r'{{fileDirname}}/../../../data/GC_genome196_train_subset500k.jsonl',
        image_folder=r'/home/pirenjie/data/vg_1.2',
        template_file=r'{{fileDirname}}/template/REC.json',
    ),
    recvg_subset=dict(
        type='RECDataset',
        filename=r'{{fileDirname}}/../../../data/GC_genome196_train_subset100.jsonl',
        image_folder=r'/home/pirenjie/data/vg_1.2',
        template_file=r'{{fileDirname}}/template/REC.json',
    ),
    gc=dict(
        type='GCDataset',
        filename=r'{{fileDirname}}/../../../data/GC_genome196_train.jsonl',
        image_folder=r'/home/pirenjie/data/vg_1.2',
        template_file=r'{{fileDirname}}/template/GC.json',
    ),
    caption=dict(
        type='CaptionDataset',
        filename=r'{{fileDirname}}/../../../data/CAP_coco2014_train.jsonl',
        image_folder=r'/home/pirenjie/data/refcoco/images/train2014',
        template_file=r'{{fileDirname}}/template/image_cap.json',
    ),
    llavacc3m=dict(
        type='InstructDataset',
        filename=r"{{fileDirname}}/../../../data/llava_cc3m.jsonl",
        image_folder=r'/home/pirenjie/data/cc3m-llava/images_595k',  # TODO: zz make folder name mistake
    ),
    llavalcs=dict(
        type='InstructDataset',
        filename=r"{{fileDirname}}/../../../data/blip_laion_cc_sbu_558k.jsonl",
        # image_folder=r'sh41:s3://MultiModal/Monolith/academic/llava-pretrain/data/595K_imgs',  # TODO: zz make folder name mistake
        image_folder=r'/home/pirenjie/data/cc3m-llava/images_558k',  # TODO: zz make folder name mistake
    ),
    instruct=dict(
        type='InstructDataset',
        filename=r'{{fileDirname}}/../../../data/llava_instruct_150k.jsonl',
        image_folder=r'/home/pirenjie/data/refcoco/images/train2014',
        add_coco_prefix=True,
    ),
    **_base_.DEFAULT_TRAIN_GQA_VARIANT,
    **_base_.DEFAULT_TRAIN_CLEVR_VARIANT,
    **_base_.DEFAULT_TRAIN_POINT_VARIANT,
    **_base_.DEFAULT_TRAIN_GPTGEN_VARIANT,
    **_base_.DEFAULT_TRAIN_VCR_VARIANT,
    **_base_.DEFAULT_TRAIN_VQAv2_VARIANT,
    **_base_.DEFAULT_TRAIN_VQAEX_VARIANT,
)
