import json
import pickle
from PIL import Image

# with open("/home/pirenjie/data/shikra/REC_ref3_train.jsonl", "r") as f:
#     objects1= list(f)
# with open("/home/pirenjie/data/refcoco/shikra_mask/anns/refcoco/train_mask.jsonl", "r") as f:
#     objects2 = list(f)
# # with open("/home/pirenjie/data/shikra/blip_laion_cc_sbu_558k.jsonl", "r") as f:
# #     objects_train = list(f)
# #
# # with open("/home/pirenjie/data/refcoco/refcoco/refs(unc).p", "rb") as f:
# #     object_refcoco = pickle.load(f)
# objects_test = [json.loads(o) for o in objects_test]
# objects_test = [o for o in objects2]
# objects_train = [json.loads(o) for o in objects_train]
# train_paths=[o["img_path"] for o in objects_train]
# test_paths=[o["img_path"] for o in objects_test]
# uniqe_test = list(set(test_paths).difference(train_paths))
with open("/home/pirenjie/data/refcoco/shikra_mask/anns/refcocog/train_mask.jsonl", "r") as f:
    objects1= list(f)
# mask_path = "/home/pirenjie/data/refcoco/shikra_mask/masks/refcoco/5911.png"
# mask = Image.open(mask_path).convert("L")
pass