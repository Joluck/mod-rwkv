
<h1 align="center">
  <p>ModRWKV: Transformer Multimodality in Linear Time</p>
</h1>
<p align="center">
        📖 <a href="">中文</a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/ZoomFly">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://arxiv.org/abs/2505.14505">Paper <b>(EMNLP'25 Oral)</b></a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://discord.com/invite/bDSBUMeFpc">Discord</a>


</p>

## Introduction
Our goal is to implement training and inference in any modality using pure **linear model** architecture. 

## Building Env
- Clone repo and direct to target DIR
```
git clone https://github.com/JL-er/WorldRWKV.git
cd WorldRWKV
```
- Dependencies
```
conda create -n world python=3.12
conda activate world
pip install -r requirements.txt #for Chinese User please add -i https://pypi.tuna.tsinghua.edu.cn/simple
# Recommend torch=>2.4.0
```
## Inference
```pyhton
python -m web.visual_web
```
#### This is also compatible with AMD graphics cards
#### If you are RX6000 series, please change the ```--offload-arch=gfx1100``` to ```--offload-arch=gfx1030``` at line 38,47,217 in ```/home/alic-li/WorldRWKV/infer/rwkv/model.py```, One-click operation for RX7000 series
#### It is assumed that you already know how to build AMD's ROCm environment~

> [!NOTE]
> Please make sure encoder model matchs encoder_type. More details are here:  world/world_encoder.py
```
from infer.worldmodel import Worldinfer
from PIL import Image


llm_path='/home/rwkv/model/rwkv7-3b-siglip/rwkv-0'
encoder_path='/home/rwkv/model/siglip2basep16s384'
encoder_type='siglip' #[clip, whisper, siglip, speech]

model = Worldinfer(model_path=llm_path, encoder_type=encoder_type, encoder_path=encoder_path)

img_path = './docs/03-Confusing-Pictures.jpg'
image = Image.open(img_path).convert('RGB')

text = '\x16User: What is unusual about this image?\x17Assistant:'

result = model.generate(text, image)

print(result)
```


## Benchmarking

We adopt [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) as our benchmark suite and implement a custom branch. It's loaded here as a submodule. Refer to [Quickstart](third_party/VLMEvalKit/docs/en/Quickstart.md) for more details.

An example usage is as follows, you will need to modify the model path in [config.json](eval/vlmevalkit/config.json)
```bash
git submodule update --init --recursive # To obtain the submodule
export PYTHONPATH=$PYTHONPATH:$(pwd)
pip install -e third_party/VLMEvalKit
python third_party/VLMEvalKit/run.py  --work-dir ./results/ --config eval/vlmevalkit/config.json
```
Currenty multi-GPU is not tested.
<Directory to save results>


## Training
> [!NOTE]
> Encoder model has to match encoder type while different tasks use different data types。You can register your own modality class in world/world_encoder.py
```
load_model=/home/rwkvos/model/rwkv/RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth
proj_dir=/home/rwkvos/peter/out_model/rwkv7-3b-pretrain-siglip
data_file=/home/rwkvos/data/hf-imgs/pretrain595

n_layer=32
n_embd=2560

encoder_path="google/siglip2-base-patch16-384" #chose your own encoder model
encoder_type=siglip # Register encoder model in worldencoder
data_type=arrow

micro_bsz=32
epoch_save=1
epoch_steps=18605 
ctx_len=2048


HF_ENDPOINT="https://hf-mirror.com" python world_train.py \   # 中国用户使用"https://hf-mirror.com"下载模型
--load_model $load_model \
--proj_dir $proj_dir --data_file $data_file \
--data_type $data_type \
--vocab_size 65536 \
--n_layer $n_layer --n_embd $n_embd \
--ctx_len $ctx_len --micro_bsz $micro_bsz \
--epoch_steps $epoch_steps --epoch_count 1 --epoch_begin 0 --epoch_save $epoch_save \
--lr_init 1e-3 --lr_final 0 --warmup_steps 0 \
--accelerator gpu --devices 8 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
--encoder_path $encoder_path --encoder_type $encoder_type \
--my_testing "x070" --train_step proj rwkv #train_step 选择你要训练的部分，proj、rwkv
```

## Hugging Face Export

This repo provides a Hugging Face export path for the Qwen3.5-based visual encoder, the custom ModRWKV processor, and a trained ModRWKV VLM checkpoint.

### 1. Export the ViT bundle

`preprocess/export_qwen35_enc.py` exports the Qwen3.5 vision encoder that is used by the multimodal wrapper.

```bash
python preprocess/export_qwen35_enc.py \
  --model_path Qwen/Qwen3.5-0.8B \
  --output_path vision_bundle \
  --max_image_tokens 4096
```

This writes a Hugging Face-compatible vision bundle into `vision_bundle/`.

### 2. Export the processor bundle

`wrapper/export_processor.py` exports the tokenizer, chat template, and image processor assets required by `AutoProcessor`.

```bash
python wrapper/export_processor.py \
  --output-dir processor_bundle \
  --image-processor Qwen/Qwen3.5-0.8B \
  --max-image-tokens 4096 \
  --force
```

This writes the processor files into `processor_bundle/` and runs a small load test after export.

### 3. Export the trained VLM bundle

`export_hf_model.py` combines:

- the trained RWKV checkpoint
- the exported vision bundle
- the exported processor bundle
- the remote-code wrapper files

into a single Hugging Face directory that can be loaded with `trust_remote_code=True`.

```bash
python export_hf_model.py \
  --checkpoint rwkv7-0.4b-sft-qwen3_5/rwkv-step-1024.pth \
  --vision-bundle vision_bundle \
  --processor-bundle processor_bundle \
  --output-dir hf_export_test \
  --overwrite
```

The exported directory contains model weights, `config.json`, processor assets, and the Python files needed for remote-code loading.

### 4. Load the exported VLM

For a visual-language model, prefer `AutoModelForImageTextToText` instead of `AutoModel`.

```python
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

bundle_path = "hf_export_test"

processor = AutoProcessor.from_pretrained(bundle_path, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    bundle_path,
    trust_remote_code=True,
).to("cuda")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": Image.open("docs/03-Confusing-Pictures.jpg").convert("RGB")},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}

generated = model.generate(
    **inputs,
    max_new_tokens=64,
    do_sample=False,
    use_cache=True,
)

print(processor.batch_decode(generated, skip_special_tokens=False)[0])
```

Use `AutoModelForCausalLM` if you specifically want the causal-LM class, but `AutoModelForImageTextToText` is the intended multimodal entry point.

### 5. Smoke test the exported bundle

You can validate an exported bundle with:

```bash
python wrapper/test_exported_model.py \
  --bundle-path hf_export_test \
  --device cuda
```


## Web-demo (Using Gradio)
```
python audio_multiturns_web.py # For Audio QA and ASR
 
python visual_web.py  # For Visual QA 

```
## Abilities
### Tasks WorldRWKV already accomplished and future direction
| Already      | Future |
|:--------------:|:-----------:|
| asr            | ✅          |
| speech to text | ✅          |
| visual to text | ✅          |
| text to speech | ❌          |
| text to visual | ❌          |
|speech to speech| ❌          |


## Visual QA Benchmarks

| **Encoder** | **LLM** | **VQAV2** | **TextVQA** | **GQA** | **ScienceQA** |**POPE**| **Checkpoint** |
|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|
| [**Clip**](https://huggingface.co/openai/clip-vit-large-patch14-336)    | RWKV7-0.4B     | 62.04      | 31.72      | 49.32       |   51.10         |
|| RWKV7-1.5B     | 72.31       | 40.27       | 54.56       |   62.77          |
|             | RWKV7-3B       | 73.13       | 45.56       | 57.00       | 70.06       |
| [**SigLIP2**](https://huggingface.co/google/siglip2-base-patch16-384) | RWKV7-0.4B|    72.04     | 38.75       | 55.52       | 43.32       |86.6|[WorldRWKV/RWKV7-0.4B-siglip2](https://huggingface.co/WorldRWKV/RWKV7-0.4B-siglip2)     |
|             | RWKV7-1.5B   |     76.95    | 44.96       | 58.88       | 63.10       |86.7|[WorldRWKV/RWKV7-1.5B-siglip2](https://huggingface.co/WorldRWKV/RWKV7-1.5B-siglip2)     |
|             | RWKV7-3B      |     78.30     |   51.09          |   60.75          |     70.93        |87.1|[WorldRWKV/RWKV7-3B-siglip2](https://huggingface.co/WorldRWKV/RWKV7-3B-siglip2)       |

## ASR Benchmarks

| **Encoder** | **LLM** | **LibriSpeech** | **Aishell-1** |
|:--------------:|:--------------:|:--------------:|:--------------:|
|[**wavlm large**](https://huggingface.co/microsoft/wavlm-large) | RWKV7-0.4B | 2.43%(clean) | 9.68%(dev) |
|            |            | 6.51%(other) | 10.33%(test) |
|[**wavlm base+**](https://huggingface.co/microsoft/wavlm-base-plus) | RWKV7-0.4B | 3.08%(clean) | 12.40%(dev) |
|            |            | 10.38%(other) | 13.46%(test) |
|[**whisper medium**](https://huggingface.co/openai/whisper-medium) | RWKV7-0.4B | 5.33%(clean) | 5.08%(dev) |
|            |            | 12.28%(other) | 5.83%(test) |
|[**whisper small**](https://huggingface.co/openai/whisper-small) | RWKV7-0.4B | 6.24%(clean) | 6.29%(dev) |
|            |            | 16.92%(other) | 6.95%(test) |

## ASR & AUDIO QA (Demo)
| **Encoder** | **LLM** | **task** | **Checkpoint** |
|:--------------:|:--------------:|:--------------:|:--------------:|
|[**wavlm large**](https://huggingface.co/microsoft/wavlm-large) | RWKV7-0.1B | EN asr|[WorldRWKV/RWKV7-0.1B-wavlmLarge-ENASR-demo](https://huggingface.co/WorldRWKV/RWKV7-0.1B-wavlmLarge-ENASR-demo)|
|            |     RWKV7-0.4B       | EN asr|[WorldRWKV/RWKV7-0.4B-wavlmLarge-ENASR-demo](https://huggingface.co/WorldRWKV/RWKV7-0.4B-wavlmLarge-ENASR-demo)|
|            |     RWKV7-0.4B       | CN asr|[WorldRWKV/RWKV7-0.4B-wavlmLarge-CNASR-demo](https://huggingface.co/WorldRWKV/RWKV7-0.4B-wavlmLarge-CNASR-demo)|
|            |     RWKV7-0.4B       | EN qa|[WorldRWKV/RWKV7-0.4B-wavlmLarge-ENQA-demo](https://huggingface.co/WorldRWKV/RWKV7-0.4B-wavlmLarge-ENQA-demo)|


## ASR Comparison

We conduct a comparative analysis of our World-RWKV model against several state-of-the-art ASR models using benchmark datasets. The results demonstrate that World-RWKV exhibits remarkable and competitive performance despite limited training steps and data. This can be attributed to its inherent potential in audio comprehension, which enables it to excel in various audio-related tasks.

### Librispeech

|**Model** | **Training Details** | **test-clean(%)** | **test-other(%)** |
|:--------------:|:--------------:|:--------------:|:--------------:|
|**WorldRWKV** | trained on 960h data with 2 epoches (about 4.4k steps) | 2.43 | 6.51 |
|**Zipformer** | trained on 960h data with 170 epoches (about 1600k steps) | 2.00 | 4.30 |
|**Paraformer-v2** | not provided | 3.00 | 6.90 |
|**SenseVoice** | trianed on private 400,000 hours of multilingual audio data | 2.57 | 4.28 |

### Aishell-1

|**Model** | **Training Details** | **test(%)** | **dev(%)** |
|:--------------:|:--------------:|:--------------:|:--------------:|
|**WorldRWKV** | trained on 170h data with 3 epoches (about 5.6k steps) | 5.83 | 5.08 | 
|**Zipformer** | trained on 170h data with 56 epoches (about 220k steps) | 4.28 | 4.03 |
|**Paraformer-v2** | not provided | 4.70 | 4.30 |
|**SenseVoice** | trianed on private 400,000 hours of multilingual audio data | 2.09 | - |
