import unittest


class TestDataHandler(unittest.TestCase):
    def setUp(self):
        self.data_handler = DataHandler()
        
    def test_load_data(self):
        self.assertRaises(ValueError, self.data_handler.load_data, 1)
    
    def test_prepared_data(self):
        self.assertRaises(ValueError, self.data_handler.prepared_data, 1)
    
    def test_plot_histograms(self):
        self.assertRaises(ValueError, self.data_handler.plot_histograms, 1)
    
    def test_plot_correlation_matrix(self):
        self.assertRaises(ValueError, self.data_handler.plot_correlation_matrix, 1)
    
    def test_plot_feature_relationships(self):
        self.assertRaises(ValueError, self.data_handler.plot_feature_relationships, 1,2)
    
    def test_versioned_data(self):
        self.assertRaises(ValueError, self.data_handler.versioned_data, 1)


class TestModelTraining(unittest.TestCase):
    def setUp(self):
        self.model_training = ModelTraining()
    
    def test_split_data(self):
        self.assertRaises(ValueError, self.model_training.split_data, 1,2,3,4)
    
    def test_get_best_model(self):
        self.assertRaises(ValueError, self.model_training.get_best_model, 1,2,3,4,5)
    
    def test_evaluation_classification_model(self):
        self.assertRaises(ValueError, self.model_training.evaluate_classification_model, 1,2,3)
    
    def test_plot_confusion_matrix(self):
        self.assertRaises(ValueError, self.model_training.plot_confusion_matrix, 1,2)
    
    def test_plot_feature_importance(self):
        self.assertRaises(ValueError, self.model_training.plot_feature_importance, 1,2,3)


class TestModelEvaluation(unittest.TestCase):
    def setUp(self):
        self.model_evaluation = ModelEvaluation()
    
    def test_train_and_evaluate_models(self):
        self.assertRaises(ValueError, self.model_evaluation.train_and_evaluate_models, 1,2,3,4,5,6)
    
    def test_train_and_evaluate_models_with_balancing(self):
        self.assertRaises(ValueError, self.model_evaluation.train_and_evaluate_models_with_balancing, 1, 2, 3, 4, 5, 6)
        
        
unittest.main(argv=[''], verbosity=2, exit=False)