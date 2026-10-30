"""Official CheXOne model-card usage (V3 oracle; no visevi import).

Qwen2_5_VLForConditionalGeneration + AutoProcessor(max_pixels=512*512), qwen_vl_utils.process_vision_info,
model.generate with the shipped generation_config (do_sample=True, temperature=1e-6, repetition_penalty=1.05).
Requests (prompt text, optional crop) are read from a JSON list so both sides ask the identical question.
"""
import argparse, json
from pathlib import Path
import torch
from PIL import Image

p = argparse.ArgumentParser()
for a in ('--requests', '--weights', '--out'):
    p.add_argument(a, type=Path, required=True)
p.add_argument('--device', default='cpu')
args = p.parse_args()
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(args.weights), torch_dtype=torch.bfloat16,
                                                           attn_implementation='sdpa').eval().to(args.device)
processor = AutoProcessor.from_pretrained(str(args.weights), max_pixels=512 * 512)
out = []
for req in json.loads(args.requests.read_text()):
    content = []
    for item in req['content']:
        if item['type'] == 'image':
            if item.get('crop'):
                with Image.open(req['image_path']) as im:
                    content.append({'type': 'image', 'image': im.convert('RGB').crop(tuple(int(v) for v in item['crop']))})
            else:
                content.append({'type': 'image', 'image': 'file://' + req['image_path']})
        else:
            content.append({'type': 'text', 'text': item['text']})
    messages = [{'role': 'user', 'content': content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors='pt').to(args.device)
    torch.manual_seed(0)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=req.get('max_new_tokens', 128), output_logits=True,
                             return_dict_in_generate=True)
    ids = gen.sequences[0, inputs['input_ids'].shape[1]:]
    probs = gen.logits[0][0].float().softmax(-1)
    opt = {l: float(probs[processor.tokenizer.encode(l, add_special_tokens=False)[0]]) for l in req.get('option_letters') or []}
    out.append({'key': req['key'], 'text': processor.batch_decode([ids], skip_special_tokens=True)[0].strip(),
                'token_ids': ids.tolist(), 'option_probs': opt})
args.out.write_text(json.dumps(out, indent=1))
print('done')
