import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from loss import BCEFocalLoss
import torch

class MyCallback(Callback):

    def __init__(self, config, datamodule):
        super(MyCallback, self).__init__()
        self.config = config
        self.datamodule = datamodule

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if pl_module.hparams.noisetuneornot:
            model = pl_module.model
            for name, para in model.named_parameters ():
                model.state_dict()[name][:] += (torch.rand(para.size()).cuda()-0.5) * self.config.noisetune_lambda * torch.std(para)
            print("  Model NoiseTune Finished!  ".center(50, '='))


    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        now = pl_module.current_epoch + 1
        if now % 50 == 0:
            trainer.save_checkpoint(f"model_ckpt/epoch{now}/"+self.config.wandb_run+f"_epoch{now}.ckpt")