from mllm.dataset.utils.io import read_img_general
from mllm.dataset.utils import Expand2square
from mllm.conversation import Conversation, get_conv_template
from mllm.dataset.process_function.shikra_process_function import *
from functools import partial
transform = Expand2square()
conv = partial(get_conv_template, name='vicuna_v1.1')


def add_image_to_input(question):
    image_token_len = 256
    replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
    question = question.replace(DEFAULT_IMAGE_TOKEN, replace_token)
    question = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {question} ASSISTANT: "
    return question

def process_input(image_path, question, tokenizer=None, image_processor=None):
    image = read_img_general(image_path)
    image, _ = transform(image)
    # construct question, add image tokens
    question = add_image_to_input(question)
    # tokenize question
    if tokenizer is not None:
        inputs = tokenizer([question], return_tensors="pt")
        input_ids = torch.as_tensor(inputs.input_ids).cuda()
    else:
        input_ids, attention_mask = None, None
    if image_processor is not None:
        image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze(0).cuda()
    return image, input_ids


if __name__ == "__main__":
    process_input(image_path="debug/images/refridgerator.png", question="In the image <image>, I need to find the refridgeator and know its coordinates. Can you please help?")
    pass