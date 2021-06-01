from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


def model_factory(model_class):
    def build_model(hparams):
        return model_class(**hparams)
    return build_model
    

available_models = {'rf': model_factory(RandomForestRegressor),
                    'linear_regression': model_factory(LinearRegression),
                    'dt': model_factory(DecisionTreeRegressor)}


def get(model_string):
    return available_models[model_string]


if __name__ == '__main__':
    
    lr_factory = get('linear_regression')
    lr_hparams = {'fit_intercept': True, 'normalize': False}
    
    lr = lr_factory(lr_hparams)
    print(lr)
    
    
    rf_factory = get('rf')
    rf_hparams = {'n_estimators': 40, 'max_depth': 3}
    
    rf = rf_factory(rf_hparams)
    print(rf)
