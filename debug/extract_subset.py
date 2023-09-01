import jsonlines
objs = []
# with jsonlines.open("/home/pirenjie/data/shikra/GC_genome196_train.jsonl") as reader, jsonlines.open("/home/pirenjie/data/shikra/GC_genome196_train_subset500k.jsonl", mode="w") as writer:
#     for i, obj in enumerate(reader):
#         objs.append(obj)
#         if i<500000:
#             writer.write(obj)
#         else:
#             break
with jsonlines.open("/home/pirenjie/data/refcoco/shikra_mask/anns/refcocog/train_mask.jsonl") as reader, jsonlines.open("/home/pirenjie/data/refcoco/shikra_mask/anns/refcocog/train_mask_subset.jsonl", mode="w") as writer:
    for i, obj in enumerate(reader):
        if i<50:
            writer.write(obj)
        else:
            break