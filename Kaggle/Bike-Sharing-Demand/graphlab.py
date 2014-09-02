import graphlab as gl
import math

# load training data
training_sframe = gl.SFrame.read_csv('train.csv')
# predict on test data
test_sframe = gl.SFrame.read_csv('test.csv')
# train a model
features = ['datetime', 'season', 'holiday', 'workingday', 'weather',
            'temp', 'atemp', 'humidity', 'windspeed']

from datetime import datetime
date_format_str = '%Y-%m-%d %H:%M:%S'

def parse_date(date_str):
    """Return parsed datetime tuple"""
    d = datetime.strptime(date_str, date_format_str)
    return {'year': d.year, 'month': d.month, 'day': d.day,
            'hour': d.hour, 'weekday': d.weekday()}

def process_date_column(data_sframe):
    """Split the 'datetime' column of a given sframe"""
    parsed_date = data_sframe['datetime'].apply(parse_date).unpack(column_name_prefix='')
    for col in ['year', 'month', 'day', 'hour', 'weekday']:
        data_sframe[col] = parsed_date[col]

process_date_column(training_sframe)
process_date_column(test_sframe)


# Create three new columns: log-casual, log-registered, and log-count
for col in ['casual', 'registered', 'count']:
    training_sframe['log-' + col] = training_sframe[col].apply(lambda x: math.log(1 + x))

m = gl.boosted_trees.create(training_sframe,
                            features=features, 
                            target='count', objective='regression',
                            num_iterations=2500)


# prediction = m.predict(test_sframe)

def make_submission(prediction, filename='submission.txt'):
    with open(filename, 'w') as f:
        f.write('datetime,count\n')
        submission_strings = test_sframe['datetime'] + ',' + prediction.astype(str)
        for row in submission_strings:
            f.write(row + '\n')



new_features = features + ['year', 'month', 'weekday', 'hour']
new_features.remove('datetime')

m1 = gl.boosted_trees.create(training_sframe,
                             features=new_features,
                             target='log-casual')

m2 = gl.boosted_trees.create(training_sframe,
                             features=new_features,
                             target='log-registered')

def fused_predict(m1, m2, test_sframe):
    """
    Fused the prediction of two separately trained models.
    The input models are trained in the log domain.
    Return the combine predictions in the original domain.
    """
    p1 = m1.predict(test_sframe).apply(lambda x: math.exp(x)-1)
    p2 = m2.predict(test_sframe).apply(lambda x: math.exp(x)-1)
    return (p1 + p2).apply(lambda x: x if x > 0 else 0)

# prediction = fused_predict(m1, m2, test_sframe)
# make_submission(prediction, 'submission.csv')

env = gl.deploy.environment.Local('hyperparam_search')
training = training_sframe[training_sframe['day'] <= 16]
validation = training_sframe[training_sframe['day'] > 16]
training.save('/tmp/training')
validation.save('/tmp/validation')


ntrees = 500
search_space = {
    'params': {
        'max_depth': [10, 15, 20],
        'min_child_weight': [5, 10, 20],
        'step_size': 0.05
    },
    'num_iterations': ntrees
}

def parameter_search(training_url, validation_url, default_params):
    """
    Return the optimal parameters in the given search space.
    The parameter returned has the lowest validation rmse.
    """
    job = gl.toolkits.model_parameter_search(env, gl.boosted_trees.create,
                                             train_set_path=training_url,
                                             save_path='/tmp/job_output',
                                             standard_model_params=default_params,
                                             hyper_params=search_space,
                                             test_set_path=validation_url)


    # When the job is done, the result is stored in an SFrame
    # The result contains attributes of the models in the search space
    # and the validation error in RMSE. 
    result = gl.SFrame('/tmp/job_output').sort('rmse', ascending=True)

    # Return the parameters with the lowest validation error. 
    optimal_params = result[['max_depth', 'min_child_weight']][0]
    optimal_rmse = result['rmse'][0]
    print 'Optimal parameters: %s' % str(optimal_params)
    print 'RMSE: %s' % str(optimal_rmse)
    return optimal_params



fixed_params = {'features': new_features,
                'verbose': False}

fixed_params['target'] = 'log-casual'
params_log_casual = parameter_search('/tmp/training',
                                     '/tmp/validation',
                                     fixed_params)

fixed_params['target'] = 'log-registered'
params_log_registered = parameter_search('/tmp/training',
                                         '/tmp/validation',
                                         fixed_params)
m_log_registered = gl.boosted_trees.create(training_sframe,
                                           features=new_features,
                                           target='log-registered',
                                           num_iterations=ntrees,
                                           params=params_log_registered,
                                           verbose=False)

m_log_casual = gl.boosted_trees.create(training_sframe,
                                       features=new_features,
                                       target='log-casual',
                                       num_iterations=ntrees,
                                       params=params_log_casual,
                                       verbose=False)

final_prediction = fused_predict(m_log_registered, m_log_casual, test_sframe)

make_submission(final_prediction, 'submission2.csv')
