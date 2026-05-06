"""
LLM wrapper for Qwen2.5-7B-Instruct
Supports both transformers (local) and vllm (faster, recommended for repeated inference)
"""

import logging
import torch

logger = logging.getLogger(__name__)


class QwenLLM:
    """
    Wrapper around Qwen2.5-7B-Instruct.
    
    use_vllm=True  → faster, needs `pip install vllm`, GPU recommended
    use_vllm=False → standard transformers, works on CPU/GPU
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        use_vllm: bool = False,
        device: str = "auto",
        temperature: float = 0.1,   # low temp for structured reasoning
        load_in_4bit: bool = False,  # enable for < 16GB VRAM
    ):
        self.model_name = model_name
        self.use_vllm = use_vllm
        self.temperature = temperature

        # ── Token usage counters (reset per feature/query via reset_token_counter) ──
        self._total_input_tokens:  int = 0
        self._total_output_tokens: int = 0
        self._total_llm_calls:     int = 0

        if use_vllm:
            self._init_vllm()
        else:
            self._init_transformers(device, load_in_4bit)

    # ── Initialization ──

    def _init_vllm(self):
        from vllm import LLM, SamplingParams
        logger.info(f"Loading {self.model_name} with vllm...")
        self.model = LLM(
            model=self.model_name,
            max_model_len=32768,   # increased from 8192 to support full grammar TOC in prompt
            dtype="bfloat16",
        )
        self._SamplingParams = SamplingParams
        logger.info("vllm model loaded.")

    def _init_transformers(self, device: str, load_in_4bit: bool):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info(f"Loading {self.model_name} with transformers...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16}
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            kwargs["device_map"] = "auto"
        elif device == "auto":
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = device

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        self.model.eval()
        logger.info("Transformers model loaded.")

    # ── Token counter API ──

    def reset_token_counter(self) -> None:
        """Reset per-feature/per-query token counters to zero."""
        self._total_input_tokens  = 0
        self._total_output_tokens = 0
        self._total_llm_calls     = 0

    def get_token_counts(self) -> dict:
        """
        Return accumulated token usage since the last reset_token_counter() call.

        Returns:
            {
              "input_tokens":  int,   # prompt tokens consumed
              "output_tokens": int,   # new tokens generated
              "total_tokens":  int,   # sum
              "llm_calls":     int,   # number of generate() calls
            }
        """
        return {
            "input_tokens":  self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "total_tokens":  self._total_input_tokens + self._total_output_tokens,
            "llm_calls":     self._total_llm_calls,
        }

    # ── Generation ──

    def generate(self, prompt: str, max_new_tokens: int = 512, json_mode: bool = True) -> str:
        """
        Generate a response. The prompt is treated as the user turn.
        Returns the assistant's response as a string.

        json_mode=True  (default): system prompt instructs the model to respond
                        ONLY in valid JSON. Used for all structured tool calls,
                        feature conclusions, and audit steps.
        json_mode=False: system prompt allows free-form prose. Used by the
                        answer_query retry path that explicitly asks for
                        plain paragraphs instead of JSON.
        """
        if json_mode:
            system_content = (
                "You are an expert linguistic typologist. "
                "Always respond in valid JSON as instructed. "
                "Do not add any text outside the JSON block."
            )
        else:
            system_content = (
                "You are an expert linguistic typologist. "
                "Provide clear, well-reasoned prose answers grounded in the evidence given."
            )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]

        if self.use_vllm:
            return self._generate_vllm(messages, max_new_tokens)
        else:
            return self._generate_transformers(messages, max_new_tokens)

    def _generate_vllm(self, messages: list[dict], max_new_tokens: int) -> str:
        from vllm import SamplingParams
        # Apply chat template manually
        text = self.model.get_tokenizer().apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        params = SamplingParams(
            temperature=self.temperature,
            max_tokens=max_new_tokens,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        outputs = self.model.generate([text], params)
        result  = outputs[0].outputs[0].text.strip()

        # ── Count tokens ──
        # prompt_token_ids is populated by vllm when include_stop_str_in_output
        # is not set; fall back to re-tokenising if absent.
        if outputs[0].prompt_token_ids is not None:
            n_input = len(outputs[0].prompt_token_ids)
        else:
            n_input = len(self.model.get_tokenizer().encode(text))
        n_output = len(outputs[0].outputs[0].token_ids)
        self._total_input_tokens  += n_input
        self._total_output_tokens += n_output
        self._total_llm_calls     += 1

        return result

    def _generate_transformers(self, messages: list[dict], max_new_tokens: int) -> str:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]

        # ── Count tokens ──
        n_input  = inputs["input_ids"].shape[1]
        n_output = new_tokens.shape[0]
        self._total_input_tokens  += n_input
        self._total_output_tokens += n_output
        self._total_llm_calls     += 1

        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # ── Batch generation (for alignment, more efficient) ──

    def batch_generate(self, prompts: list[str], max_new_tokens: int = 256) -> list[str]:
        """Generate responses for multiple prompts at once (vllm only for true batching)."""
        if self.use_vllm:
            from vllm import SamplingParams
            tokenizer = self.model.get_tokenizer()
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in prompts
            ]
            params = SamplingParams(temperature=self.temperature, max_tokens=max_new_tokens)
            outputs = self.model.generate(texts, params)
            return [o.outputs[0].text.strip() for o in outputs]
        else:
            # Sequential fallback
            return [self.generate(p, max_new_tokens) for p in prompts]