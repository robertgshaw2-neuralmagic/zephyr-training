import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from contextlib import contextmanager

from sparseml.core.framework import Framework
import sparseml.core.session as session_manager
from sparseml.transformers.sparsification.obcq.export import _reload_model_state

# model_id = "/home/mgoin/models/Nous-Hermes-llama-2-7b-pruned50-quant-pt"

class SparseAutoModelForCausalLM(AutoModelForCausalLM):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        print("loading model")
        model = super().from_pretrained(*args, **kwargs)

        model_id = args[0]
        recipe_file = model_id + "/recipe.yaml"

        original_state_dict = model.state_dict()

        print("loading manager")
        session_manager.create_session()
        session_manager.pre_initialize_structure(
            model=model,
            recipe=recipe_file,
            framework=Framework.pytorch,
        )

        # reload the state dict for the model now that architecture matches expected
        print("reloading model")
        _reload_model_state(model, model_id, original_state_dict)

        return model


# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = SparseAutoModelForCausalLM.from_pretrained(
#     model_id,
#     device_map="cpu",
#     torch_dtype=torch.float32,
#     trust_remote_code=True,
# )

# prompt = "What are the main challenges to support a long context for LLM?"
# input_ids = tokenizer.encode(prompt, return_tensors="pt")
# outputs = model.generate(input_ids, max_new_tokens=100)
# print(tokenizer.decode(outputs[0]))