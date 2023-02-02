import warnings
import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from callbacks import MyCallback
from transformers import BertTokenizer, logging
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from data_module import ASTEDataModule
from module import ASTEModule

warnings.filterwarnings("ignore")
logging.set_verbosity_error()


def train(args):
    pl.seed_everything(args.seed, workers=True)
    wandb_logger = WandbLogger(project=args.wandb_project,
                               name=args.wandb_run,
                               log_model=False)
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_tokens(["n't"])

    pl_module = ASTEModule(args)
    # pl_module = ASTEModule.load_from_checkpoint('model_ckpt/epoch100/14lap_Jan18_11.08.20_ori_epoch100.ckpt')
    wandb_logger.watch(pl_module)
    pl_data_module = ASTEDataModule(args, tokenizer)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_tri_f1_for_monitor',
        mode='max',
        dirpath=f"{args.output_path}/{args.dataset}" ,
        filename=args.wandb_run + '_{val_tri_f1:.5f}_{epoch}',
        save_top_k= 3 if not args.NoFocalLoss else 2,
        # save_top_k=0,
        save_last=False,
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = args.wandb_run + '_{epoch}_last'

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    callbacks_store = [

        checkpoint_callback, lr_monitor,

        MyCallback(args, pl_data_module),

        # EarlyStopping(
        #     monitor=args.monitor_var,
        #     mode=args.opti_mode,
        #     min_delta=args.min_delta,
        #     patience=args.patience)
    ]


    trainer = pl.Trainer(
        gpus=args.gpus,
        num_sanity_val_steps=args.num_sanity_val_steps,
        callbacks=callbacks_store,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        max_epochs=args.max_epochs,
        logger=wandb_logger
    )
    trainer.fit(pl_module, datamodule=pl_data_module)

    trainer.test(pl_module, datamodule=pl_data_module)


@hydra.main(config_path='./config', config_name='main')
def main(conf):
    train(conf)


if __name__ == '__main__':
    main()
