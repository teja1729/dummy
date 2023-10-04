from fastapi import FastAPI

import logging

# Lit-GPT imports
import sys
import time
from pathlib import Path
import huggingface_hub as hf
hf.login(token="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", write_permission=True)
#import json
import gc

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

#import lightning as L
import torch

torch.set_float32_matmul_precision("high")

# from lit_gpt import GPT, Tokenizer, Config
# from lit_gpt.utils import lazy_load, quantization

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# Toy submission imports
from helper import toysubmission_generate
from api import (
    ProcessRequest,
    ProcessResponse,
    TokenizeRequest,
    TokenizeResponse,
    Token,
)

app = FastAPI()

logger = logging.getLogger(__name__)
# Configure the logging module
logging.basicConfig(level=logging.INFO)

#quantize = "bnb.nf4-dq"  # 4-bit NormalFloat with Double-Quantization (see QLoRA paper)
checkpoint_dir = "Supersaiyan1729/LLM-Science-Exam-gpt2"
#precision = "bf16-true"  # weights and data in bfloat16 precision

#fabric = L.Fabric(devices=1, accelerator="cuda", precision=precision)

#teja This is for loading the model
# with open(checkpoint_dir / "lit_config.json") as fp:
#     config = Config(**json.load(fp))

# checkpoint_path = checkpoint_dir / "lit_model.pth"
# logger.info(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")
# with fabric.init_module(empty_init=True), quantization(quantize):
#     model = GPT(config)

# with lazy_load(checkpoint_path) as checkpoint:
#     model.load_state_dict(checkpoint, strict=quantize is None)
model = AutoPeftModelForCausalLM.from_pretrained(checkpoint_dir)



device = "cuda"

model.eval()
#model = fabric.setup(model)
model.to("cuda")

#tokenizer = Tokenizer(checkpoint_dir)

tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir,trust_remote_code = True)


@app.post("/process")
async def process_request(input_data: ProcessRequest) -> ProcessResponse:
    logger.info("Using device: {}".format(device))
 
    encoded = torch.tensor(tokenizer(input_data.prompt)['input_ids'],device=device)
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + input_data.max_new_tokens

    t0 = time.perf_counter()
    tokens, logprobs, top_logprobs = toysubmission_generate(
        model,
        encoded,
        max_returned_tokens,
        max_seq_length=max_returned_tokens,
        temperature=input_data.temperature,
        top_k=input_data.top_k,
    )

    t = time.perf_counter() - t0

    #model.reset_cache()
    gc.collect()
    if input_data.echo_prompt is False:
        output = tokenizer.decode(tokens[prompt_length:])
    else:
        output = tokenizer.decode(tokens)
    tokens_generated = tokens.size(0) - prompt_length
    logger.info(
        f"Time for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec"
    )

    logger.info(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
    generated_tokens = []
    for t, lp, tlp in zip(tokens, logprobs, top_logprobs):
        idx, val = tlp
        tok_str = tokenizer.decode([idx])
        token_tlp = {tok_str: val}
        generated_tokens.append(
            Token(text=tokenizer.decode(t), logprob=lp, top_logprob=token_tlp)
        )
    logprobs_sum = sum(logprobs)
    # Process the input data here
    return ProcessResponse(
        text=output, tokens=generated_tokens, logprob=logprobs_sum, request_time=t
    )


@app.post("/tokenize")
async def tokenize(input_data: TokenizeRequest) -> TokenizeResponse:
    logger.info("Using device: {}".format(device))
    t0 = time.perf_counter()
    # encoded = tokenizer.encode(
    #     input_data.text, bos=True, eos=False, device=device
    # )
    encoded = torch.tensor(tokenizer(input_data.text)['input_ids'],device=device)
    t = time.perf_counter() - t0
    tokens = encoded.tolist()
    return TokenizeResponse(tokens=tokens, request_time=t)
