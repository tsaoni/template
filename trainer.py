import random
import torch
import logging
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import Dataset, DataLoader

def get_logger(filename=None):
    if filename is not None:
        logging.basicConfig(filename=filename, level=logging.DEBUG)
        logger = logging.getLogger()
    else: # log to console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logger.addHandler(console_handler)

    """
    # Write some log messages
    logger.debug('This is a debug message')
    logger.info('This is an info message')
    logger.warning('This is a warning message')
    logger.error('This is an error message')
    logger.critical('This is a critical message')
    """

    return logger

class pseudo_dataset(Dataset):
    def __init__(self, **kwargs):
        self.data_num = kwargs['data_num']
        self.data_dim = kwargs['data_dim']
        self.class_dim = kwargs['class_dim']
        self.generate_random_data()

    def generate_random_data(self):
        self.data, self.label = [], []
        for i in range(self.data_num):
            self.data.append([float(random.randint(0, 1))] * self.data_dim)
            self.label.append([random.randint(0, self.class_dim - 1)])

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        return {'data': self.data[idx], 'labels': self.label[idx]}

    def collate_fn(self, batch):
        batch_data, batch_label = [], []
        for b in batch:
            batch_data.append(b['data'])
            batch_label.append(b['labels'])
        return torch.Tensor(batch_data), torch.Tensor(batch_label)

class BinaryClassifier(pl.LightningModule):

    logger = get_logger()
    on_train_counter = 0
    training_counter = 0
    log_metric = 'val_acc'

    """ setting """

    def __init__(
            self, 
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # dataset config
        self.dataset_kwargs: dict = dict(
            train=dict(
                data_num=self.hparams.train_data_num,
                data_dim=self.hparams.data_dim,
                class_dim=self.hparams.class_dim,
            ),
            valid=dict(
                data_num=self.hparams.valid_data_num,
                data_dim=self.hparams.data_dim,
                class_dim=self.hparams.class_dim,
            ),
            test=dict(
                data_num=self.hparams.test_data_num,
                data_dim=self.hparams.data_dim,
                class_dim=self.hparams.class_dim,
            ),
        )
        #self.batch_size = kwargs.batch_size

        # model config
        #self.input_dim = input_dim
        #self.hidden_dim = hidden_dim
        #self.output_dim = output_dim
        self.fc1 = torch.nn.Linear(self.hparams.input_dim, self.hparams.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hparams.hidden_dim, self.hparams.output_dim)

        # other config
        #self.log_dir = log_dir

    def setup(self, mode='train', stage=None):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    """ dataset and dataloader """

    def _get_dataset(self, mode):
        mode_dataset_kwargs = self.dataset_kwargs[mode]
        dataset = pseudo_dataset(**mode_dataset_kwargs)
        return dataset

    def _get_dataloader(self, mode: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = self._get_dataset(mode)
        dataloader = DataLoader(dataset, batch_size=batch_size, \
                collate_fn=dataset.collate_fn, shuffle=shuffle)
        return dataloader

    def train_dataloader(self):
        return self._get_dataloader('train', self.hparams.batch_size, shuffle=True)

    def val_dataloader(self):
        return self._get_dataloader('valid', self.hparams.batch_size, shuffle=False)

    def test_dataloader(self):
        return self._get_dataloader('test', self.hparams.batch_size, shuffle=False)

    """ step """

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        preds = y_hat > 0.5
        loss = F.binary_cross_entropy(y_hat, y)
        acc = torch.sum(preds == y).item() / (len(y) * 1.0)
        self.log('train_loss', loss)
        return {'loss': loss, 'acc': acc}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        preds = y_hat > 0.5
        acc = torch.sum(preds == y).item() / (len(y) * 1.0)
        self.log('val_acc', acc)
        return {'val_loss': loss, 'val_acc': acc}
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        preds = y_hat > 0.5
        acc = torch.sum(preds == y).item() / (len(y) * 1.0)
        self.log('test_acc', acc)
        return {'test_loss': loss, 'test_acc': acc}

    """ logging and postprocessing """

    def epoch_end(self, epoch, result):
        """ deprecated function"""
        pass

    def training_epoch_end(self, output):
        self.training_counter += 1 

    def validation_epoch_end(self, outputs):
        self.training_counter += 1 
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        metrics_dict = {'val_epoch_end_loss': avg_loss, }
        return metrics_dict

    def on_train_epoch_end(self):
        self.on_train_counter += 1

    """ save and load """

    """
    def on_save_checkpoint(self, checkpoint):
        # 99% of use cases you don't need to implement this method
        checkpoint['something_cool_i_want_to_save'] = my_cool_pickable_object
    """

    """
    def on_load_checkpoint(self, checkpoint):
        # 99% of the time you don't need to implement this method
        self.something_cool_i_want_to_save = checkpoint['something_cool_i_want_to_save']
    """

def unuse():
    raise NotImplementedError

if __name__ == '__main__':

    train_data_num = 500
    valid_data_num = 100
    test_data_num = 100
    data_dim = 50
    class_dim = 2

    """
    train_dataset = pseudo_dataset(train_data_num, data_dim, class_dim)
    val_dataset = pseudo_dataset(valid_data_num, data_dim, class_dim)
    """

    batch_size = 32

    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, \
            collate_fn = train_dataset.collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, \
            collate_fn = val_dataset.collate_fn)
    """

    input_dim = data_dim
    hidden_dim = 5
    output_dim = class_dim - 1
    epoch_num = 3
    log_dir = '.'

    model = BinaryClassifier(log_dir=log_dir, input_dim=input_dim, hidden_dim=hidden_dim, \
            output_dim=output_dim, train_data_num=train_data_num, valid_data_num=valid_data_num, \
            test_data_num=test_data_num, data_dim=data_dim, class_dim=class_dim, batch_size=batch_size)

    early_stop_callback = EarlyStopping(monitor='val_loss')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss')
    trainer = pl.Trainer(max_epochs=epoch_num, callbacks=[early_stop_callback, checkpoint_callback])
    # trainer.fit(model, train_loader, val_loader)
    trainer.fit(model)
    val_loss = trainer.logged_metrics['val_loss'] # get metric return from logs

    print(f"on train epoch: {model.on_train_counter}")
    print(f"training epoch: {model.training_counter}")
