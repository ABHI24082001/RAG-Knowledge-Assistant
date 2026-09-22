import logging
import threading
from typing import Any

from app.config import get_settings

NO_ANSWER = "I could not find that information in the indexed documents."
logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self) -> None:
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.device: str | None = None
        self._torch: Any | None = None
        self._load_lock = threading.Lock()

    def _load_runtime(self) -> None:
        if self._torch is None:
            import torch

            self._torch = torch

    def _select_device(self) -> str:
        self._load_runtime()
        configured_device = get_settings().rag_llm_device
        if configured_device == "mps" and self._torch.backends.mps.is_available():
            return "mps"
        if configured_device == "cuda" and self._torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_model(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        with self._load_lock:
            if self.model is not None and self.tokenizer is not None:
                return
            self.device = self._select_device()
            settings = get_settings()
            logger.info("Loading local LLM %s on %s", settings.llm_model_name, self.device)
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(settings.llm_model_name)
                self.model = AutoModelForCausalLM.from_pretrained(settings.llm_model_name)
                self.model.to(self.device)
                self.model.eval()
            except Exception as exc:
                self.tokenizer = None
                self.model = None
                raise RuntimeError(f"Could not load local LLM: {exc}") from exc
            logger.info("Local LLM loaded successfully: %s", settings.llm_model_name)

    def generate_answer(self, question: str, context: str) -> str:
        self._load_model()
        messages = [
            {"role": "system", "content": (
                "You are a knowledge assistant. Use only the provided context. "
                "Do not invent facts. If the context does not contain enough "
                f"information, say exactly: {NO_ANSWER} "
                "Give a concise answer and do not mention internal embeddings, "
                "vectors, or Qdrant unless asked."
            )},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"{messages[0]['content']}\n\n{messages[1]['content']}\n\nAnswer:"
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            with self._torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            answer_tokens = output[0][inputs["input_ids"].shape[-1] :]
            return self.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip() or NO_ANSWER
        except Exception as exc:
            raise RuntimeError(f"Could not generate local LLM answer: {exc}") from exc


llm_service = LLMService()
