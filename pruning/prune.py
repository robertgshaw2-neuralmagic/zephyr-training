import torch, tqdm, argparse, os
from transformers import AutoModelForCausalLM
import sparseml.core.session as session_manager
from sparseml.core.framework import Framework
from sparseml.optim.helpers import load_recipe_yaml_str
from sparseml.transformers.data.base_llm import TransformersDataset

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--recipe-path", type=str, required=True)
parser.add_argument("--save-path", type=str, required=True)

DATASET_ID = "HuggingFaceH4/ultrachat_200k"
SEQUENCE_LENGTH = 512
NSAMPLES = 512

class ChatDataset(TransformersDataset):
    def __init__(
        self,
        model,
        seqlen,
        nsamples,
        path,
        seed: int = 0,
        split: str = "train_sft",
        split_percent_to_use: float = 1.0,
    ):
        super().__init__(
            model=model,
            seqlen=seqlen,
            nsamples=nsamples,
            path=path,
            name=None,
            seed=seed,
            split=split,
            use_max_tokens=False,
            split_percent_to_use=split_percent_to_use,
        )

        system_prompt = {
            "content": "You are a friendly chatbot",
            "role": "system"
        }

        processed_data = []
        for sample in tqdm.tqdm(self._data):
            assert "messages" in sample
            messages_with_sys_prompt = [system_prompt] + sample["messages"]
            processed_data.append(
                self.tokenizer.apply_chat_template(
                    messages_with_sys_prompt, 
                    tokenize=False
                )
            )
            
        self.create_dataloader(processed_data)

def run_obcq(model_path, recipe_path, save_path):
    
    print("\n---------- Loading Dataset ----------")
    dataset = ChatDataset(
        model=model_path,
        seqlen=SEQUENCE_LENGTH,
        nsamples=NSAMPLES,
        path=DATASET_ID,
    )
    calibration_data = dataset.loader
    tokenizer = dataset.tokenizer

    print("\n---------- Loading Model ----------")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16)

    print("\n---------- Running Pruning ----------")
    session_manager.create_session()
    session = session_manager.active_session()
    session.apply(
        framework=Framework.pytorch,
        recipe=recipe_path,
        model=model,
        calib_data=calibration_data,
        start=0.0,
        device="cuda",
        copy_data=False,
    )

    print("\n---------- Saving Model ----------")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    recipe_output_path = os.path.join(save_path, "recipe.yaml")
    with open(recipe_output_path, "w") as fp:
        fp.write(load_recipe_yaml_str(recipe_path))
    print(f"Saved to: {save_path}")

    
if __name__ == "__main__":
    args = parser.parse_args()
    run_obcq(
        model_path=args.model_path,
        recipe_path=args.recipe_path,
        save_path=args.save_path
    )

    