from transformers import Trainer # type: ignore
from torch.nn import BCEWithLogitsLoss # type: ignore


class CustomTrainerForMultilabelClassification(Trainer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):

        inputs["labels"] = inputs["labels"].float() 
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]
        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss