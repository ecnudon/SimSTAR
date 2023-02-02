import pytorch_lightning as pl
import numpy as np
from loss import *
from metrics import metrics
from torch.optim import AdamW
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup
)
from model import SimSTAR
import json

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule,
    "constant_w_warmup": get_constant_schedule_with_warmup,
}


class ASTEModule(pl.LightningModule):
    def __init__(self, args):
        super(ASTEModule, self).__init__()
        self.save_hyperparameters(args)
        self.model = SimSTAR(args)

        # I add an extra token `n't` in BertTokenizer, thus,
        self.model.resize_token_embeddings(30523)
        # Due to the saving and reloading of checkpoint in pytorch_lightning
        # restrains the MyModule.__init__() with only one param. (i.e. args)
        # So I have to place the ugly code here. Sorry for that.

        self.metrics = metrics(args)
        if self.hparams.NoFocalLoss:
            self.loss_fn = basic_loss()
        else:
            self.loss_fn = BCEFocalLoss(gamma=self.hparams.gamma, alpha=self.hparams.alpha,
                                    tri_contribution=self.hparams.tricon, reduction='sum')



    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if
                           not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
                "name": "normal_params"
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "name": "no_decay_params"
            },
        ]
        count = 0
        for dict in optimizer_grouped_parameters:
            count += sum([p.nelement() for p in dict['params']])
        print(count)

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate)

        lr_scheduler = self._get_lr_scheduler(self.hparams.max_steps, optimizer)


        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'epoch'}]

    def _get_lr_scheduler(self, num_training_steps, optimizer):
        schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        if self.hparams.lr_scheduler == "constant":
            scheduler = schedule_func(optimizer)
        elif self.hparams.lr_scheduler == "constant_w_warmup":
            scheduler = schedule_func(optimizer, num_warmup_steps=self.hparams.warmup_steps)
        elif self.hparams.lr_scheduler == "inverse_square_root":
            scheduler = schedule_func(optimizer, num_warmup_steps=self.hparams.warmup_steps)
        elif self.hparams.lr_scheduler == "cosine_w_restarts":
            scheduler = schedule_func(optimizer, num_warmup_steps=self.hparams.warmup_steps,
                                      num_training_steps=num_training_steps, num_cycles=5)
        else:
            scheduler = schedule_func(
                optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=num_training_steps)
        return scheduler

    def training_step(self, batch, batch_idx):
        data, labels = batch.values()
        term_pred, senti_pred = self.model(**data)
        term_gold, senti_gold = labels.values()
        loss = self.loss_fn(term_pred, term_gold, senti_pred, senti_gold)

        # dict_loss = {'Train_term_loss': term_loss, 'Train_senti_loss':senti_loss}
        # self.log_dict(dict_loss, on_epoch=True)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def validation_step(self, batch, batch_idx):

        data, labels = batch.values()
        term_pred, senti_pred = self.model(**data)
        term_gold, senti_gold = labels.values()
        loss = self.loss_fn(term_pred, term_gold, senti_pred, senti_gold)

        term_count = self.metrics.count_term_num(term_pred, term_gold)
        senti_count = self.metrics.count_tri_num(term_pred, term_gold, senti_pred, senti_gold)

        loss = {'loss': loss }
        self.logging_with_prefix(loss, prefix='val')

        return term_count, senti_count

    def test_step(self, batch, batch_idx):
        data, labels = batch.values()
        term_pred, senti_pred = self.model(**data)
        term_gold, senti_gold = labels.values()
        term_count = self.metrics.count_term_num(term_pred, term_gold)
        senti_count, return_display = self.metrics.count_tri_num_and_display(
            term_pred, term_gold, senti_pred, senti_gold, data['original_text'])


        return term_count, senti_count, return_display


    def validation_epoch_end(self, outputs):
        term_result, senti_result = [0] * len(outputs[0][0]), [0] * len(outputs[0][1])
        for output in outputs:
            term_result = np.sum([term_result, output[0]], axis=0)
            senti_result = np.sum([senti_result, output[1]], axis=0)

        values = {}
        term_prf = self.metrics.calculate(*term_result)
        for num, num_type in zip(term_prf, self.metrics.value_targets):
            values['term' + num_type] = num
        senti_prf = self.metrics.calculate(*senti_result)
        for num, num_type in zip(senti_prf, self.metrics.value_targets):
            values['tri' + num_type] = num

        self.logging_with_prefix(values, prefix='val')

        if self.current_epoch > self.hparams.epoch_start_monitoring:
            self.log('val_tri_f1_for_monitor', max(values['tri_f1']-0.5, 0))
        else:
            self.log('val_tri_f1_for_monitor', 0)


    def test_epoch_end(self, outputs):
        term_result, senti_result = [0] * len(outputs[0][0]), [0] * len(outputs[0][1])
        all_tri_list = []
        for output in outputs:
            term_result = np.sum([term_result, output[0]], axis=0)
            senti_result = np.sum([senti_result, output[1]], axis=0)
            all_tri_list += output[2]

        values = {}
        term_prf = self.metrics.calculate(*term_result)
        for num, num_type in zip(term_prf, self.metrics.value_targets):
            values['term' + num_type] = num
        senti_prf = self.metrics.calculate(*senti_result)
        for num, num_type in zip(senti_prf, self.metrics.value_targets):
            values['tri' + num_type] = num

        self.logging_with_prefix(values, prefix='test')

        json_str = json.dumps(all_tri_list, indent=2)
        # with open(f'{self.hparams.timestamp}_test.json', 'w') as json_file:
        with open(f'_test_output_temp.json', 'w') as json_file:
            json_file.write(json_str)


    def logging_with_prefix(self, log_dict, prefix='val'):
        to_log = {}
        for key, value in log_dict.items():
            to_log[prefix+'_'+key] = value
        self.log_dict(to_log, on_epoch=True)

    def forward(self, data):
        return self.model.forward(**data)