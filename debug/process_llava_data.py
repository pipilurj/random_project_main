import jsonlines
with jsonlines.open("/home/pirenjie/data/shikra/blip_laion_cc_sbu_558k.jsonl") as reader, jsonlines.open("/home/pirenjie/data/shikra/blip_laion_cc_sbu_558k_processed.jsonl", mode="w") as writer:
    for i, obj in enumerate(reader):
        obj['image'] = "GCC_train_" + obj['image'].split("/")[1]
        writer.write(obj)
