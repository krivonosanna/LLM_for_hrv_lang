import torch
import torch.nn as nn
from torch import Tensor
from tqdm.auto import tqdm, trange


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    assert num_training_steps >= num_warmup_steps

    def lr_lambda(current_step):
        if current_step <= num_warmup_steps:
            return current_step / num_warmup_steps

        return (num_training_steps - current_step) / (num_training_steps - num_warmup_steps)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def cross_entropy_loss(input_ids: Tensor, attention_mask: Tensor, logits: Tensor) -> Tensor:
    B, L, V = logits.shape
    input_ids = input_ids[:, 1:]
    logits = logits[:, :-1]

    input_ids = input_ids.flatten()
    logits = logits.reshape(-1, V)
    loss = nn.CrossEntropyLoss(reduction='none')(logits, input_ids)
    loss = loss.reshape(B, L - 1) * attention_mask[:, 1:]
    loss = loss.sum() / attention_mask[:, 1:].sum()

    return loss

class Trainer:
    def __init__(
        self,
        config,
        logger
    ):
        self.learning_rate = float(config["trainer"]["learning_rate"])
        self.weight_decay = config["trainer"]["weight_decay"]
        self.clip_grad_norm = config["trainer"]["clip_grad_norm"]
        self.n_steps = config["trainer"]["n_steps"]
        self.val_every_n_steps = config["trainer"]["val_every_n_steps"]
        self.plot_every_n_steps = config["trainer"]["plot_every_n_steps"]

        self.logger = logger

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print("running on device", self.device)

    @torch.no_grad()
    def validate(self, model, val_loader):
        model.eval()
        val_loss = 0.0
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            input_ids, attention_mask = batch
            input_ids = input_ids.to(self.device, non_blocking=True)
            attention_mask = attention_mask.to(self.device, non_blocking=True)

            logits = model(input_ids, attention_mask)  # [bs; seq len; vocab size]
            val_loss += cross_entropy_loss(input_ids, attention_mask, logits)
        return val_loss / len(val_loader)

    def run(self, model, train_loader, val_loader=None):
        model = model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0.1 * self.n_steps, num_training_steps=self.n_steps
        )
        model.train()

        #plotlosses = PlotLosses(figsize=(15, 9), step_names="Step")
        logs = {"lr": 0, "epoch": 0}

        data_iter = iter(train_loader)
        for iter_num in range(self.n_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                logs["epoch"] += 1
                batch = next(data_iter)

            input_ids, attention_mask = batch
            input_ids = input_ids.to(self.device, non_blocking=True)
            attention_mask = attention_mask.to(self.device, non_blocking=True)

            logits = model(input_ids, attention_mask)  # [bs; seq len; vocab size]
            loss = cross_entropy_loss(input_ids, attention_mask, logits)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
            optimizer.step()
            scheduler.step()

            if val_loader is not None and iter_num > 0 and iter_num % self.val_every_n_steps == 0:
                val_loss = self.validate(model, val_loader)
                self.logger.info(f"Epoch {logs['epoch']}, Step {iter_num}: val_loss = {val_loss.item():.4f}")
                # plotlosses.update({"val_loss": val_loss.item()}, current_step=iter_num)
                # plotlosses.send()
                model.train()

            if iter_num % self.plot_every_n_steps == 0:
                logs["loss"] = loss.item()
                logs["lr"] = scheduler.get_last_lr()[0]
                self.logger.info(f"Epoch {logs['epoch']}, Step {iter_num}: train_loss = {loss.item():.4f}")
                # plotlosses.update(logs, current_step=iter_num)
                # plotlosses.send()
        if val_loader is not None:
            val_loss = self.validate(model, val_loader)
            self.logger.info(f"final val_loss = {val_loss.item():.4f}")
            # plotlosses.update({"val_loss": val_loss.item()}, current_step=iter_num)
            # plotlosses.send()
