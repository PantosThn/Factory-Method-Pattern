from sklearn.pipeline import Pipeline
import preprocessing


def build_pipeline(preprocessing_steps, model):
    return Pipeline(preprocessing.get_pipeline_steps(preprocessing_steps) + [('model', model)])
