"""
Regression Model Builder

Class responsible for building different types of regression models.
This class encapsulates the model building functionality from the original modules.
"""

import os
from typing import Any, Dict, List
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.regression import (
    LinearRegression, LinearRegressionModel,
    RandomForestRegressor, RandomForestRegressionModel,
    GBTRegressor, GBTRegressionModel,
    DecisionTreeRegressor, DecisionTreeRegressionModel,
    FMRegressor, FMRegressionModel
)

# Advanced ML algorithms (optional imports with fallbacks)
try:
    from xgboost.spark import SparkXGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️ XGBoost not available for regression. Install with: pip install xgboost>=1.6.0")
    XGBOOST_AVAILABLE = False

try:
    from synapse.ml.lightgbm import LightGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("⚠️ LightGBM not available for regression. Install with: pip install synapseml>=0.11.0")
    LIGHTGBM_AVAILABLE = False


def linear_regression_model(train, x, y):
    """Build Linear Regression model."""
    lr = LinearRegression(
        featuresCol=x,
        labelCol=y,
        regParam=0.01,
        elasticNetParam=0.0
    )
    lrModel = lr.fit(train)
    return lrModel


def random_forest_regression_model(train, x, y):
    """Build Random Forest Regression model."""
    rf = RandomForestRegressor(
        featuresCol=x,
        labelCol=y,
        numTrees=100,
        maxDepth=5,
        maxBins=256,  # Increased to handle very high-cardinality categorical features (up to 256 unique values)
        seed=42
    )
    rfModel = rf.fit(train)
    return rfModel


def gradient_boosting_regression_model(train, x, y):
    """Build Gradient Boosting Regression model with optimizations to reduce large task binary warnings."""
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
    
    gbt = GBTRegressor(
        featuresCol=x,
        labelCol=y,
        maxIter=100,
        maxDepth=5,
        maxBins=128,  # Reduced from 256 to 128 to reduce task binary size
        seed=42
    )
    gbtModel = gbt.fit(train)
    return gbtModel


def decision_tree_regression_model(train, x, y):
    """Build Decision Tree Regression model."""
    dt = DecisionTreeRegressor(
        featuresCol=x,
        labelCol=y,
        maxDepth=5,
        maxBins=256,  # Increased to handle very high-cardinality categorical features (up to 256 unique values)
        seed=42
    )
    dtModel = dt.fit(train)
    return dtModel


def fm_regression_model(train, x, y):
    """Build Factorization Machine Regression model."""
    fm = FMRegressor(
        featuresCol=x,
        labelCol=y,
        factorSize=8,
        seed=42
    )
    fmModel = fm.fit(train)
    return fmModel


def xgboost_regression_model(train, x, y):
    """Build XGBoost Regression model."""
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not available. Install with: pip install xgboost>=1.6.0")
    
    xgb = SparkXGBRegressor(
        features_col=x,  # Fixed: Use features_col instead of featuresCol
        label_col=y,     # Fixed: Use label_col instead of labelCol
        maxDepth=6,
        eta=0.3,
        numRound=100,
        seed=42
    )
    xgbModel = xgb.fit(train)
    return xgbModel


def lightgbm_regression_model(train, x, y):
    """Build LightGBM Regression model."""
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM is not available. Install with: pip install synapseml>=0.11.0")
    
    lgb = LightGBMRegressor(
        featuresCol=x,
        labelCol=y,
        numLeaves=31,
        learningRate=0.1,
        numIterations=100,
        seed=42
    )
    lgbModel = lgb.fit(train)
    return lgbModel


class RegressionModelBuilder:
    """
    Model builder class that handles building different types of regression models.
    
    This class provides functionality for:
    - Building various regression models
    - Saving and loading models
    - Model parameter configuration
    """
    
    def __init__(self, spark_session: SparkSession):
        """
        Initialize the regression model builder.
        
        Args:
            spark_session: PySpark SparkSession
        """
        self.spark = spark_session
        
        # Model type mappings
        self.model_types = {
            'linear_regression': {
                'class': LinearRegression,
                'model_class': LinearRegressionModel,
                'build_func': linear_regression_model
            },
            'random_forest': {
                'class': RandomForestRegressor,
                'model_class': RandomForestRegressionModel,
                'build_func': random_forest_regression_model
            },
            'gradient_boosting': {
                'class': GBTRegressor,
                'model_class': GBTRegressionModel,
                'build_func': gradient_boosting_regression_model
            },
            'decision_tree': {
                'class': DecisionTreeRegressor,
                'model_class': DecisionTreeRegressionModel,
                'build_func': decision_tree_regression_model
            },
            'fm_regression': {
                'class': FMRegressor,
                'model_class': FMRegressionModel,
                'build_func': fm_regression_model
            }
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.model_types['xgboost'] = {
                'class': SparkXGBRegressor,
                'model_class': None,
                'build_func': xgboost_regression_model
            }
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            self.model_types['lightgbm'] = {
                'class': LightGBMRegressor,
                'model_class': None,
                'build_func': lightgbm_regression_model
            }
        
        print(f"✅ RegressionModelBuilder initialized with {len(self.model_types)} model types")
        if XGBOOST_AVAILABLE:
            print("   📦 XGBoost regression available")
        if LIGHTGBM_AVAILABLE:
            print("   📦 LightGBM regression available")
    
    def build_model(self, train_data: DataFrame, features_col: str, 
                   label_col: str, model_type: str, **params) -> Any:
        """
        Build a regression model.
        
        Args:
            train_data: Training DataFrame
            features_col: Name of the features column
            label_col: Name of the label column
            model_type: Type of model to build
            **params: Additional model parameters for hyperparameter optimization
            
        Returns:
            Trained model
        """
        if model_type not in self.model_types:
            available_types = list(self.model_types.keys())
            raise ValueError(f"Unsupported model type: {model_type}. Available types: {available_types}")
        
        print(f"Building {model_type} regression model...")
        
        # Extract hyperparameters from params (if any)
        hyperparams = {k: v for k, v in params.items() if k not in ['num_features']}
        
        if hyperparams:
            print(f"   🔧 Using optimized parameters: {hyperparams}")
            # Build model with hyperparameters using specific functions
            model = self._build_model_with_hyperparams(train_data, features_col, label_col, model_type, hyperparams)
        else:
            # Build model using default function
            model_config = self.model_types[model_type]
            build_func = model_config['build_func']
            model = build_func(train_data, features_col, label_col)
        
        print(f"✅ {model_type} regression model built successfully")
        return model
    
    def _build_model_with_hyperparams(self, train_data: DataFrame, features_col: str, 
                                     label_col: str, model_type: str, hyperparams: Dict[str, Any]) -> Any:
        """Build a model with specific hyperparameters."""
        
        if model_type == 'linear_regression':
            lr = LinearRegression(
                featuresCol=features_col,
                labelCol=label_col,
                regParam=hyperparams.get('regParam', 0.01),
                elasticNetParam=hyperparams.get('elasticNetParam', 0.0),
                maxIter=hyperparams.get('maxIter', 100)
            )
            return lr.fit(train_data)
        
        elif model_type == 'random_forest':
            rf = RandomForestRegressor(
                featuresCol=features_col,
                labelCol=label_col,
                numTrees=hyperparams.get('numTrees', 100),
                maxDepth=hyperparams.get('maxDepth', 5),
                maxBins=hyperparams.get('maxBins', 256),
                minInstancesPerNode=hyperparams.get('minInstancesPerNode', 1),
                subsamplingRate=hyperparams.get('subsamplingRate', 1.0),
                seed=42
            )
            return rf.fit(train_data)
        
        elif model_type == 'gradient_boosting':
            # Apply gradient boosting specific optimizations
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession()
            if spark:
                try:
                    import sys
                    import os
                    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    if parent_dir not in sys.path:
                        sys.path.insert(0, parent_dir)
                    from spark_optimization_config import apply_gradient_boosting_optimizations
                    apply_gradient_boosting_optimizations(spark)
                except Exception as e:
                    print(f"⚠️ Could not apply gradient boosting optimizations: {e}")
            
            gbt = GBTRegressor(
                featuresCol=features_col,
                labelCol=label_col,
                maxIter=hyperparams.get('maxIter', 100),
                maxDepth=hyperparams.get('maxDepth', 6),
                maxBins=hyperparams.get('maxBins', 128),  # Reduced default from 256 to 128
                stepSize=hyperparams.get('stepSize', 0.1),
                subsamplingRate=hyperparams.get('subsamplingRate', 1.0),
                seed=42
            )
            return gbt.fit(train_data)
        
        elif model_type == 'decision_tree':
            dt = DecisionTreeRegressor(
                featuresCol=features_col,
                labelCol=label_col,
                maxDepth=hyperparams.get('maxDepth', 5),
                maxBins=hyperparams.get('maxBins', 256),
                minInstancesPerNode=hyperparams.get('minInstancesPerNode', 1),
                minInfoGain=hyperparams.get('minInfoGain', 0.0),
                seed=42
            )
            return dt.fit(train_data)
        
        elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
            xgb = SparkXGBRegressor(
                features_col=features_col,
                label_col=label_col,
                max_depth=hyperparams.get('max_depth', 6),
                n_estimators=hyperparams.get('n_estimators', 100),
                learning_rate=hyperparams.get('learning_rate', 0.3),
                subsample=hyperparams.get('subsample', 1.0),
                colsample_bytree=hyperparams.get('colsample_bytree', 1.0),
                min_child_weight=hyperparams.get('min_child_weight', 1),
                gamma=hyperparams.get('gamma', 0.0),
                seed=42
            )
            return xgb.fit(train_data)
        
        elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            lgb = LightGBMRegressor(
                featuresCol=features_col,
                labelCol=label_col,
                numLeaves=hyperparams.get('numLeaves', 31),
                numIterations=hyperparams.get('numIterations', 100),
                learningRate=hyperparams.get('learningRate', 0.1),
                featureFraction=hyperparams.get('featureFraction', 1.0),
                baggingFraction=hyperparams.get('baggingFraction', 1.0),
                minDataInLeaf=hyperparams.get('minDataInLeaf', 20),
                lambdaL1=hyperparams.get('lambdaL1', 0.0),
                lambdaL2=hyperparams.get('lambdaL2', 0.0),
                seed=42
            )
            return lgb.fit(train_data)
        
        else:
            # Fall back to default function if hyperparameter version not implemented
            model_config = self.model_types[model_type]
            build_func = model_config['build_func']
            return build_func(train_data, features_col, label_col)
    
    def save_model(self, model: Any, path: str):
        """
        Save a trained model.
        
        Args:
            model: Trained model to save
            path: Path to save the model
        """
        os.makedirs(path, exist_ok=True)
        model.write().overwrite().save(path)
        print(f"✅ Regression model saved to {path}")
    
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
            available_types = list(self.model_types.keys())
            raise ValueError(f"Unsupported model type: {model_type}. Available types: {available_types}")
        
        model_config = self.model_types[model_type]
        model_class = model_config['model_class']
        
        if model_class:
            model = model_class.load(path)
        else:
            # For XGBoost and LightGBM, use generic loading
            from pyspark.ml import PipelineModel
            model = PipelineModel.load(path)
        
        print(f"✅ {model_type} regression model loaded from {path}")
        return model
    
    def validate_model_type(self, model_type: str) -> bool:
        """
        Validate if a model type is supported.
        
        Args:
            model_type: Type of model to validate
            
        Returns:
            True if model type is supported, False otherwise
        """
        return model_type in self.model_types
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available model types.
        
        Returns:
            List of available model type names
        """
        return list(self.model_types.keys()) 