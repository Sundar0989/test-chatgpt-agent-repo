"""
Model Builder

Class responsible for building different types of classification models.
This class encapsulates the model building functionality from the original modules.
"""

import os
from typing import Dict, Any, List
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.classification import (
    LogisticRegression, LogisticRegressionModel,
    RandomForestClassifier, RandomForestClassificationModel,
    GBTClassifier, GBTClassificationModel,
    DecisionTreeClassifier, DecisionTreeClassificationModel,
    MultilayerPerceptronClassifier, MultilayerPerceptronClassificationModel
)
import joblib

# Advanced ML algorithms (optional imports with fallbacks)
try:
    from xgboost.spark import SparkXGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️ XGBoost not available. Install with: pip install xgboost>=1.6.0")
    XGBOOST_AVAILABLE = False

try:
    from synapse.ml.lightgbm import LightGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("⚠️ LightGBM not available. Install with: pip install synapseml>=0.11.0")
    LIGHTGBM_AVAILABLE = False

def logistic_model(train, x, y):
    lr = LogisticRegression(featuresCol = x, labelCol = y, maxIter = 10)
    lrModel = lr.fit(train)
    return lrModel

def randomForest_model(train, x, y):
    rf = RandomForestClassifier(featuresCol = x, labelCol = y, numTrees=10)
    rfModel = rf.fit(train)
    return rfModel

def gradientBoosting_model(train, x, y):
    """Build Gradient Boosting Classification model with optimizations to reduce large task binary warnings."""
    # Apply gradient boosting specific optimizations to reduce broadcasting warnings
    from pyspark.sql import SparkSession
    spark = SparkSession.getActiveSession()
    if spark:
        try:
            # Import and apply optimizations
            import sys
            import os
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            from spark_optimization_config import apply_gradient_boosting_optimizations
            apply_gradient_boosting_optimizations(spark)
        except Exception as e:
            print(f"⚠️ Could not apply gradient boosting optimizations: {e}")
    
    gb = GBTClassifier(
        featuresCol=x, 
        labelCol=y, 
        maxIter=100,    # Increased from 10 for better performance
        maxDepth=6,     # Added explicit maxDepth
        maxBins=128     # Reduced from default 256 to reduce task binary size
    )
    gbModel = gb.fit(train)
    return gbModel

def decisionTree_model(train, x, y):
    dt = DecisionTreeClassifier(featuresCol = x, labelCol = y, maxDepth=5)
    dtModel = dt.fit(train)
    return dtModel

def neuralNetwork_model(train, x, y, feature_count, num_classes):
    layers = [feature_count, feature_count*3, feature_count*2, num_classes]
    mlp = MultilayerPerceptronClassifier(featuresCol = x, labelCol = y, maxIter=100, layers=layers, blockSize=512, seed=12345)
    mlpModel = mlp.fit(train)
    return mlpModel

def xgboost_model(train, x, y, num_classes=2):
    """Build XGBoost model using Spark XGBoost integration."""
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not available. Install with: pip install xgboost>=1.6.0")
    
    # Configure XGBoost parameters
    # Note: SparkXGBClassifier automatically determines objective based on label data
    xgb = SparkXGBClassifier(
        features_col=x,
        label_col=y,
        max_depth=6,
        n_estimators=100,
        num_workers=1,
        use_gpu=False
    )
    xgbModel = xgb.fit(train)
    return xgbModel

def lightgbm_model(train, x, y, num_classes=2):
    """Build LightGBM model using SynapseML integration."""
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM not available. Install with: pip install synapseml>=0.11.0")
    
    # Configure LightGBM parameters
    if num_classes > 2:
        # Multi-class classification
        lgb = LightGBMClassifier(
            featuresCol=x,
            labelCol=y,
            objective="multiclass",
            numLeaves=31,
            numIterations=100,
            learningRate=0.1
        )
        # lgb = lgb.setNumClass(num_classes)
    else:
        # Binary classification
        lgb = LightGBMClassifier(
            featuresCol=x,
            labelCol=y,
            objective="binary",
            numLeaves=31,
            numIterations=100,
            learningRate=0.1
        )
    
    lgbModel = lgb.fit(train)
    return lgbModel

class ModelBuilder:
    """
    Model builder class that handles building different types of classification models.
    
    This class provides functionality for:
    - Building various classification models
    - Saving and loading models
    - Model parameter configuration
    """
    
    def __init__(self, spark_session: SparkSession):
        """
        Initialize the model builder.
        
        Args:
            spark_session: PySpark SparkSession
        """
        self.spark = spark_session
        
        # Model type mappings
        self.model_types = {
            'logistic': {
                'class': LogisticRegression,
                'model_class': LogisticRegressionModel,
                'build_func': logistic_model
            },
            'random_forest': {
                'class': RandomForestClassifier,
                'model_class': RandomForestClassificationModel,
                'build_func': randomForest_model
            },
            'gradient_boosting': {
                'class': GBTClassifier,
                'model_class': GBTClassificationModel,
                'build_func': gradientBoosting_model
            },
            'decision_tree': {
                'class': DecisionTreeClassifier,
                'model_class': DecisionTreeClassificationModel,
                'build_func': decisionTree_model
            },
            'neural_network': {
                'class': MultilayerPerceptronClassifier,
                'model_class': MultilayerPerceptronClassificationModel,
                'build_func': neuralNetwork_model
            }
        }
        
        # Add advanced models if available
        if XGBOOST_AVAILABLE:
            self.model_types['xgboost'] = {
                'class': SparkXGBClassifier,
                'model_class': None,  # XGBoost model class varies
                'build_func': xgboost_model
            }
        
        if LIGHTGBM_AVAILABLE:
            self.model_types['lightgbm'] = {
                'class': LightGBMClassifier,
                'model_class': None,  # LightGBM model class varies
                'build_func': lightgbm_model
            }
    
    def build_model(self, train_data: DataFrame, features_col: str, 
                   label_col: str, model_type: str, **params) -> Any:
        """
        Build a classification model.
        
        Args:
            train_data: Training DataFrame
            features_col: Name of the features column
            label_col: Name of the label column
            model_type: Type of model to build
            **params: Additional model parameters
            
        Returns:
            Trained model
        """
        if model_type not in self.model_types:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        print(f"Building {model_type} model...")
        
        # Get model configuration
        model_config = self.model_types[model_type]
        build_func = model_config['build_func']
        
        # Build model using the original function
        if model_type == 'neural_network':
            feature_count = params.get('feature_count')  # Extract feature_count from params
            num_classes = params.get('num_classes', 2)  # Extract num_classes from params, default to 2
            model = build_func(train_data, features_col, label_col, feature_count, num_classes)
        elif model_type in ['xgboost', 'lightgbm']:
            num_classes = params.get('num_classes', 2)  # Extract num_classes for advanced models
            model = build_func(train_data, features_col, label_col, num_classes)
        else:
            model = build_func(train_data, features_col, label_col)
        
        print(f"{model_type} model built successfully.")
        return model
    
    def create_estimator(self, features_col: str, label_col: str, model_type: str, **params) -> Any:
        """
        Create an unfitted estimator for cross-validation.
        
        Args:
            features_col: Name of the features column
            label_col: Name of the label column
            model_type: Type of model to create
            **params: Additional model parameters
            
        Returns:
            Unfitted estimator (for use with CrossValidator)
        """
        if model_type not in self.model_types:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        print(f"Creating {model_type} estimator for cross-validation...")
        
        # Create unfitted estimators
        if model_type == 'logistic':
            estimator = LogisticRegression(featuresCol=features_col, labelCol=label_col, maxIter=10)
        elif model_type == 'random_forest':
            estimator = RandomForestClassifier(featuresCol=features_col, labelCol=label_col, numTrees=10)
        elif model_type == 'gradient_boosting':
            estimator = GBTClassifier(featuresCol=features_col, labelCol=label_col, maxIter=10)
        elif model_type == 'decision_tree':
            estimator = DecisionTreeClassifier(featuresCol=features_col, labelCol=label_col, maxDepth=5)
        elif model_type == 'neural_network':
            feature_count = params.get('feature_count', 10)
            num_classes = params.get('num_classes', 2)
            layers = [feature_count, feature_count*3, feature_count*2, num_classes]
            estimator = MultilayerPerceptronClassifier(
                featuresCol=features_col, labelCol=label_col, 
                maxIter=100, layers=layers, blockSize=512, seed=12345
            )
        elif model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available. Install with: pip install xgboost>=1.6.0")
            num_classes = params.get('num_classes', 2)
            # Note: SparkXGBClassifier automatically determines objective based on label data
            estimator = SparkXGBClassifier(
                features_col=features_col,
                label_col=label_col,
                max_depth=6,
                n_estimators=100,
                num_workers=1,
                use_gpu=False
            )
        elif model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not available. Install with: pip install synapseml>=0.11.0")
            
            try:
                num_classes = params.get('num_classes', 2)
                if num_classes > 2:
                    # Multi-class classification
                    estimator = LightGBMClassifier(
                        featuresCol=features_col,
                        labelCol=label_col,
                        objective="multiclass",
                        numLeaves=31,
                        numIterations=100,
                        learningRate=0.1
                    )
                 #   estimator = estimator.setNumClass(num_classes)
                else:
                    # Binary classification
                    estimator = LightGBMClassifier(
                        featuresCol=features_col,
                        labelCol=label_col,
                        objective="binary",
                        numLeaves=31,
                        numIterations=100,
                        learningRate=0.1
                    )
            except Exception as e:
                if "'JavaPackage' object is not callable" in str(e):
                    raise ImportError(
                        "SynapseML JARs not loaded in Spark session. "
                        "Use 'from spark_optimization_config import create_optimized_spark_session; "
                        "spark = create_optimized_spark_session(include_lightgbm=True)' to create a properly configured session."
                    ) from e
                else:
                    raise e
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        print(f"{model_type} estimator created successfully.")
        return estimator

    def save_model(self, model: Any, path: str):
        """
        Save a trained model.
        
        Args:
            model: Trained model to save
            path: Path to save the model
        """
        os.makedirs(path, exist_ok=True)
        model.write().overwrite().save(path)
        print(f"Model saved to {path}")
    
    def load_model(self, model_type: str, path: str) -> Any:
        """
        Load a saved model.
        
        Args:
            model_type: Type of model to load
            path: Path to the saved model
            
        Returns:
            Loaded model
        """
        if model_type not in self.model_types:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model_config = self.model_types[model_type]
        model_class = model_config['model_class']
        
        # Handle special cases for models without standard model_class
        if model_class is None:
            if model_type == 'xgboost':
                # For XGBoost, load as standalone SparkXGBClassifierModel
                try:
                    from xgboost.spark import SparkXGBClassifierModel
                    model = SparkXGBClassifierModel.load(path)
                    print(f"Loaded {model_type} model as SparkXGBClassifierModel from {path}")
                    return model
                except Exception as e:
                    print(f"Error loading XGBoost model: {str(e)}")
                    raise
            elif model_type == 'lightgbm':
                # For LightGBM, load as standalone model
                try:
                    from synapse.ml.lightgbm import LightGBMClassificationModel
                    model = LightGBMClassificationModel.load(path)
                    print(f"Loaded {model_type} model from {path}")
                    return model
                except Exception as e:
                    print(f"Error loading LightGBM model: {str(e)}")
                    raise
            else:
                raise ValueError(f"No model class defined for {model_type}")
        else:
            # For standard Spark ML models
            model = model_class.load(path)
            print(f"Loaded {model_type} model from {path}")
            return model
    
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """
        Get information about a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary containing model information
        """
        if model_type not in self.model_types:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model_config = self.model_types[model_type]
        
        info = {
            'model_type': model_type,
            'class_name': model_config['class'].__name__,
            'model_class_name': model_config['model_class'].__name__,
            'description': self._get_model_description(model_type)
        }
        
        return info
    
    def _get_model_description(self, model_type: str) -> str:
        """
        Get description for a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Model description
        """
        descriptions = {
            'logistic': 'Logistic Regression - Linear classification model',
            'random_forest': 'Random Forest - Ensemble of decision trees',
            'gradient_boosting': 'Gradient Boosting - Sequential ensemble of weak learners',
            'decision_tree': 'Decision Tree - Tree-based classification model',
            'neural_network': 'Neural Network - Multi-layer perceptron classifier'
        }
        
        return descriptions.get(model_type, 'Unknown model type')
    
    def get_supported_models(self) -> List[str]:
        """
        Get list of supported model types.
        
        Returns:
            List of supported model types
        """
        return list(self.model_types.keys())
    
    def validate_model_type(self, model_type: str) -> bool:
        """
        Validate if a model type is supported.
        
        Args:
            model_type: Type of model to validate
            
        Returns:
            True if supported, False otherwise
        """
        return model_type in self.model_types 