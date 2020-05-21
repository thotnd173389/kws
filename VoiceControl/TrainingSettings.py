import os
class TrainingSettings(object):
    def __init__(self,
                 how_many_training_steps = '600,600',
                 learning_rate = '0.001,0.0001',
                 eval_step_interval = 400,
                 train_dir = './training/',
                 summaries_dir = './summary/logs/',
                 optimizer_epsilon = 1e-08):
        self.how_many_training_steps = how_many_training_steps
        self.learning_rate = learning_rate
        self.eval_step_interval = eval_step_interval
        self.train_dir = train_dir
        self.summaries_dir = summaries_dir
        self.optimizer_epsilon = optimizer_epsilon
        
        