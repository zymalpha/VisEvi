"""CheXOne (Qwen2.5-VL-3B) worker for A6, A7, A18 and A24.

Requests carry a list of content items (``image`` with optional crop box in
original pixels, or ``text``). Decoding follows the shipped generation
configuration. For multiple-choice prompts
the probabilities of the option letters at the first generated position are
returned as raw model scores (not calibrated probabilities).
"""
from __future__ import annotations

import os
from pathlib import Path

from ._runtime import device_runtime, serve

MAX_PIXELS = 512 * 512          # author recommendation: align with training


def build_messages(content):
    """Build the model-card conversation without adding a project system prompt."""
    return [{'role': 'user', 'content': content}]


def load(weights_dir, device):
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model, info = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        weights_dir, torch_dtype=torch.bfloat16, attn_implementation='sdpa', output_loading_info=True)
    problems = {k: v for k, v in info.items() if v}
    if problems:
        raise RuntimeError(f'checkpoint does not match the model exactly: {problems}')
    model = model.eval().to(device)
    # Match the author's model-card path: max_pixels is set, while min_pixels and
    # per-image overrides are left to the processor's saved configuration.
    processor = AutoProcessor.from_pretrained(weights_dir, max_pixels=MAX_PIXELS)
    runtime = device_runtime(device)
    runtime.update(dtype='bfloat16', attention='sdpa', max_pixels=MAX_PIXELS,
                   decoding='shipped generation_config (do_sample=%s, temperature=%s, '
                   'repetition_penalty=%s)' % (model.generation_config.do_sample,
                                                model.generation_config.temperature,
                                                model.generation_config.repetition_penalty),
                   option_probabilities='softmax of raw first-position logits')
    return (model, processor), runtime


def _image(path, crop):
    from PIL import Image
    with Image.open(path) as image:
        rgb = image.convert('RGB')
    if crop:
        rgb = rgb.crop(tuple(int(v) for v in crop))
    return rgb


def answer(model, processor, image_path, spec):
    import torch
    from qwen_vl_utils import process_vision_info

    content, sizes = [], []
    for item in spec['content']:
        if item['type'] == 'image':
            image = _image(image_path, item.get('crop'))
            sizes.append(list(image.size))
            content.append({'type': 'image', 'image': image})
        else:
            content.append({'type': 'text', 'text': item['text']})
    messages = build_messages(content)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=images, padding=True, return_tensors='pt').to(model.device)
    with torch.inference_mode():
        # Do not override the checkpoint's generation configuration.  The
        # official model-card path relies on its do_sample/temperature/
        # repetition_penalty settings; raw logits below are still reported
        # before the generation processors alter token selection.
        out = model.generate(**inputs, max_new_tokens=int(spec.get('max_new_tokens', 256)),
                             output_logits=True, return_dict_in_generate=True)
    generated = out.sequences[0, inputs['input_ids'].shape[1]:]
    reply = processor.decode(generated, skip_special_tokens=False)
    for token in ('<|im_end|>', '<|endoftext|>'):
        reply = reply.split(token)[0]
    result = {'text': reply.strip(), 'input_image_sizes_wh': sizes,
              'model_grid_thw': inputs['image_grid_thw'].tolist() if 'image_grid_thw' in inputs else None,
              'generated_tokens': int(generated.shape[0]), 'token_ids': generated.tolist()}
    options = spec.get('option_letters')
    if options:
        # raw logits of the first generated position: the model's own distribution, before the
        # shipped repetition_penalty (1.05) that penalises option letters present in the prompt
        probs = out.logits[0][0].float().softmax(-1)
        ids = {letter: processor.tokenizer.encode(letter, add_special_tokens=False) for letter in options}
        result['first_token_option_probabilities'] = {
            letter: float(probs[token_ids[0]]) for letter, token_ids in ids.items() if len(token_ids) == 1}
    return result


def handle(state, request):
    model, processor = state
    params = request['params']
    answers = [answer(model, processor, request['image_path'], spec) for spec in params['requests']]
    return {'answers': answers,
            'request_params': {k: v for k, v in params.items() if k != 'requests'}}, []


if __name__ == '__main__':
    serve(load, handle)
