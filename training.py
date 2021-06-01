import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yaml

import models
import preprocessing
import utils


class Trainer:

    def __init__(self, model_tag, hparams, preprocessing=None, tune=False):

        self.tune = tune
        self.hparams = hparams if not tune else {}
        model = models.get(model_tag)(self.hparams)

        if preprocessing:
            model = utils.build_pipeline(preprocessing, model)

        self.model = GridSearchCV(model, hparams) if tune else model

    def fit(self, x, y):

        self.model.fit(x, y)

        if self.tune:
            self.tune = False
            self.hparams = self.model.best_params_
            self.model = self.model.best_estimator_

    def predict(self, x):
        return self.model.predict(x)

    def evaluate(self, x, y, *args, **kwargs):
        preds = self.predict(x)
        return {'mse': mean_squared_error(y, preds),
                'mae': mean_absolute_error(y, preds)}

    @staticmethod
    def build_trainer_from_config(config):

        preprocess = config.get('preprocessing')

        return Trainer(config['name'], hparams=config['hparams'],
                       tune=config['tune'], preprocessing=preprocess)

    @staticmethod
    def run_experiment(config_path, x, y):  # TODO: change x, y to train_dir, test_dir

        # x_train, y_train = preprocessing.load_data(train_path)
        # x_train, y_train = preprocessing.preprocess(x_train, y_train)

        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        trainer = Trainer.build_trainer_from_config(config)

        trainer.fit(x, y)
        score = trainer.evaluate(x, y)

        with open('logs/' + config_path.split('/')[-1].replace('.yaml', '.txt'), 'w') as f:
            for k, v in score.items():
                f.write('{}   -->   {:.2f}\n'.format(k, v))

        return trainer


if __name__ == '__main__':
    x, y = preprocessing.load_data('data')
    x, y = preprocessing.preprocess(x, y)

    #     trainer = Trainer.run_experiment('config/rf_config_hparam_tuning.yaml', train_path, test_path)
    trainer = Trainer.run_experiment('config/dt_config_hparam_tuning.yaml', x, y)
