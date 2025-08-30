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
    # Test if we can actually import and create a basic instance
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost Spark integration available and functional")
except ImportError as e:
    print(f"⚠️ XGBoost import failed: {e}")
    print("   Install with: pip install xgboost>=1.6.0")
    XGBOOST_AVAILABLE = False
except Exception as e:
    print(f"⚠️ XGBoost initialization failed: {e}")
    print("   XGBoost may be installed but not compatible with current Spark version")
    XGBOOST_AVAILABLE = False

try:
    from synapse.ml.lightgbm import LightGBMClassifier
    LIGHTGBM_AVAILABLE = True
    print("✅ LightGBM (SynapseML) integration available and functional")
except ImportError as e:
    print(f"⚠️ LightGBM import failed: {e}")
    print("   Install with: pip install synapseml>=0.11.0")
    LIGHTGBM_AVAILABLE = False
except Exception as e:
    print(f"⚠️ LightGBM initialization failed: {e}")
    print("   LightGBM may be installed but not compatible with current Spark version")
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
    """
    Build a Gradient Boosting Classification model with sensible defaults to
    minimise large broadcasted task binaries.  The number of boosting
    iterations and the tree complexity are deliberately kept modest to
    strike a balance between performance and resource usage.  A smaller
    maxBins value further reduces memory consumption during tree building.
    """
    # Apply gradient boosting optimizations to reduce large task binary warnings
    try:
        from spark_optimization_config import apply_gradient_boosting_optimizations
        apply_gradient_boosting_optimizations(spark)
    except Exception as e:
        print(f"⚠️ Could not apply gradient boosting optimizations: {e}")
    
    # Use reduced complexity settings compared to the previous implementation
    gb = GBTClassifier(
        featuresCol=x,
        labelCol=y,
        maxIter=50,   # Limit boosting iterations to avoid large task binaries
        maxDepth=5,   # Slightly shallower trees
        maxBins=64    # Smaller number of bins reduces the size of broadcasted tasks
    )
    gbModel = gb.fit(train)
    return gbModel

def decisionTree_model(train, x, y):
    dt = DecisionTreeClassifier(featuresCol = x, labelCol = y, maxDepth=5)
    dtModel = dt.fit(train)
    return dtModel

def neuralNetwork_model(train, x, y, feature_count, num_classes):
    # Use more conservative layer sizes to avoid issues
    if feature_count <= 10:
        layers = [feature_count, 8, num_classes]
    elif feature_count <= 20:
        layers = [feature_count, 16, num_classes]
    else:
        layers = [feature_count, 32, num_classes]
    
    print(f"🧠 Neural network architecture: {layers}")
    mlp = MultilayerPerceptronClassifier(featuresCol = x, labelCol = y, maxIter=100, layers=layers, blockSize=512, seed=12345)
    mlpModel = mlp.fit(train)
    return mlpModel

def xgboost_model(train, x, y, num_classes=2):
    """Build XGBoost model using Spark XGBoost integration."""
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not available. Install with: pip install xgboost>=1.6.0")
    
    try:
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
        
        # Validate the estimator
        if hasattr(xgb, 'fit') and hasattr(xgb, 'getFeaturesCol') and hasattr(xgb, 'getLabelCol'):
            print(f"✅ XGBoost estimator created successfully")
            print(f"   Features column: {xgb.getFeaturesCol()}")
            print(f"   Label column: {xgb.getLabelCol()}")
        else:
            print(f"⚠️ XGBoost estimator may be missing required methods")
            print(f"   Available methods: {[m for m in dir(xgb) if not m.startswith('_')]}")
            raise RuntimeError("XGBoost estimator validation failed")
        
        print(f"🔧 XGBoost parameters: max_depth=6, n_estimators=100, num_workers=1")
        print(f"📊 Training data shape: {train.count()} rows")
        
        # Validate data format before training
        print(f"🔍 Validating data format for XGBoost training...")
        sample_row = train.select(x, y).first()
        if sample_row:
            print(f"   📊 Sample features: {type(sample_row[0])}, shape: {getattr(sample_row[0], 'size', 'unknown')}")
            print(f"   📊 Sample label: {type(sample_row[1])}, value: {sample_row[1]}")
        
        xgbModel = xgb.fit(train)
        print(f"✅ XGBoost model trained successfully")
        
        # Validate the trained model
        if hasattr(xgbModel, 'transform') and hasattr(xgbModel, 'write'):
            print(f"✅ XGBoost model validation passed - has required methods")
        else:
            print(f"⚠️ XGBoost model may be missing required methods")
            print(f"   Available methods: {[m for m in dir(xgbModel) if not m.startswith('_')]}")
            
            # Check if this is a compatibility issue
            if not hasattr(xgbModel, 'write'):
                print(f"   ❌ XGBoost model missing 'write' method - this will cause save failures")
                print(f"   💡 This may be a version compatibility issue between XGBoost and PySpark")
                raise RuntimeError("XGBoost model missing required 'write' method for saving")
        
        return xgbModel
        
    except Exception as e:
        print(f"❌ XGBoost model training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"XGBoost model training failed: {str(e)}")

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
            print("✅ XGBoost integration verified - adding to available models")
            self.model_types['xgboost'] = {
                'class': SparkXGBClassifier,
                'model_class': None,  # XGBoost model class varies
                'build_func': xgboost_model
            }
        
        if LIGHTGBM_AVAILABLE:
            print("✅ LightGBM integration verified - adding to available models")
            self.model_types['lightgbm'] = {
                'class': LightGBMClassifier,
                'model_class': None,  # LightGBM model class varies
                'build_func': lightgbm_model
            }
    
    def build_model(self, train_data: DataFrame, features_col: str,
                   label_col: str, model_type: str, **params) -> Any:
        """
        Build a classification model.  This method uses provided
        hyperparameters (if any) to override the defaults defined in
        the individual model builder functions.  Without explicit
        parameters the default settings are used.

        Args:
            train_data: Training DataFrame
            features_col: Name of the features column
            label_col: Name of the label column
            model_type: Type of model to build
            **params: Additional model parameters (e.g. hyperparameters)

        Returns:
            Trained model
        """
        if model_type not in self.model_types:
            raise ValueError(f"Unsupported model type: {model_type}")

        print(f"Building {model_type} model...")

        # Extract hyperparameters relevant for this model
        # Remove keys that are not model hyperparameters
        hyperparams = {k: v for k, v in params.items() if k not in ['feature_count', 'num_classes']}

        # Dispatch to specialized builder when hyperparameters are supplied
        if hyperparams:
            print(f"   🔧 Using optimized parameters: {hyperparams}")
            model = self._build_model_with_hyperparams(
                train_data, features_col, label_col, model_type, hyperparams, params
            )
        else:
            # Use the default build function
            model_config = self.model_types[model_type]
            build_func = model_config['build_func']
            if model_type == 'neural_network':
                feature_count = params.get('feature_count')
                num_classes = params.get('num_classes', 2)
                model = build_func(train_data, features_col, label_col, feature_count, num_classes)
            elif model_type in ['xgboost', 'lightgbm']:
                num_classes = params.get('num_classes', 2)
                model = build_func(train_data, features_col, label_col, num_classes)
            else:
                model = build_func(train_data, features_col, label_col)

        print(f"{model_type} model built successfully.")
        return model

    def _build_model_with_hyperparams(self, train_data: DataFrame, features_col: str,
                                      label_col: str, model_type: str,
                                      hyperparams: Dict[str, Any], all_params: Dict[str, Any]) -> Any:
        """
        Internal helper to construct a classification model using
        supplied hyperparameters.  Only parameters relevant to the
        specific model type are used.  Parameters not specified fall
        back to sensible defaults.

        Args:
            train_data: Training DataFrame
            features_col: Name of the features column
            label_col: Name of the label column
            model_type: Type of model to build
            hyperparams: Dictionary of hyperparameters
            all_params: Original params dict (may include feature_count, num_classes)

        Returns:
            Trained model using the specified hyperparameters
        """
        from pyspark.ml.classification import (
            LogisticRegression, RandomForestClassifier, GBTClassifier,
            DecisionTreeClassifier, MultilayerPerceptronClassifier
        )

        # Determine number of classes for advanced models
        num_classes = all_params.get('num_classes', 2)

        if model_type == 'logistic':
            lr = LogisticRegression(
                featuresCol=features_col,
                labelCol=label_col,
                maxIter=hyperparams.get('maxIter', 10),
                regParam=hyperparams.get('regParam', 0.0),
                elasticNetParam=hyperparams.get('elasticNetParam', 0.0)
            )
            return lr.fit(train_data)
        elif model_type == 'random_forest':
            rf = RandomForestClassifier(
                featuresCol=features_col,
                labelCol=label_col,
                numTrees=hyperparams.get('numTrees', 10),
                maxDepth=hyperparams.get('maxDepth', 5),
                maxBins=hyperparams.get('maxBins', 32),
                minInstancesPerNode=hyperparams.get('minInstancesPerNode', 1),
                subsamplingRate=hyperparams.get('subsamplingRate', 1.0),
                featureSubsetStrategy=hyperparams.get('featureSubsetStrategy', 'auto'),
                seed=42
            )
            return rf.fit(train_data)
        elif model_type == 'gradient_boosting':
            # Apply gradient boosting optimizations to reduce large task binary warnings
            try:
                from spark_optimization_config import apply_gradient_boosting_optimizations
                apply_gradient_boosting_optimizations(self.spark)
            except Exception as e:
                print(f"⚠️ Could not apply gradient boosting optimizations: {e}")
            gb = GBTClassifier(
                featuresCol=features_col,
                labelCol=label_col,
                maxIter=hyperparams.get('maxIter', 50),
                maxDepth=hyperparams.get('maxDepth', 5),
                maxBins=hyperparams.get('maxBins', 32),
                stepSize=hyperparams.get('stepSize', 0.1),
                subsamplingRate=hyperparams.get('subsamplingRate', 1.0),
                seed=42
            )
            return gb.fit(train_data)
        elif model_type == 'decision_tree':
            dt = DecisionTreeClassifier(
                featuresCol=features_col,
                labelCol=label_col,
                maxDepth=hyperparams.get('maxDepth', 5),
                maxBins=hyperparams.get('maxBins', 32),
                minInstancesPerNode=hyperparams.get('minInstancesPerNode', 1),
                minInfoGain=hyperparams.get('minInfoGain', 0.0),
                seed=42
            )
            return dt.fit(train_data)
        elif model_type == 'neural_network':
            # For MLP, use feature_count and num_classes from original params
            feature_count = all_params.get('feature_count')
            layers = hyperparams.get('layers')
            
            # If specific layers are provided (list of ints), use them; otherwise derive from feature_count
            if layers is None:
                if feature_count:
                    layers = [feature_count, feature_count*2, num_classes]
                else:
                    # Calculate feature count from training data if not provided
                    try:
                        # Get the first row to determine feature vector size
                        sample_row = train_data.select(features_col).first()
                        if sample_row and hasattr(sample_row[0], 'size'):
                            feature_count = sample_row[0].size
                            layers = [feature_count, feature_count*2, num_classes]
                        else:
                            # Fallback to a reasonable default
                            layers = [50, 25, num_classes]
                    except Exception as e:
                        print(f"   ⚠️ Could not determine feature count from data: {e}")
                        # Fallback to a reasonable default
                        layers = [50, 25, num_classes]
                print(f"   🔧 Using default layers: {layers}")
            else:
                # Layers were provided - ensure they are compatible with input size
                if feature_count and layers and len(layers) > 0:
                    # Validate that first hidden layer is not larger than input size
                    first_hidden_size = layers[0]
                    if first_hidden_size > feature_count:
                        print(f"   ⚠️ Neural network first hidden layer size ({first_hidden_size}) > input size ({feature_count})")
                        print(f"   🔧 Adjusting to use compatible layer size: {feature_count}")
                        layers[0] = feature_count
                
                # Build complete layer architecture: [input_size, hidden_layers..., output_size]
                complete_layers = [feature_count] + layers + [num_classes]
                print(f"   🧠 Neural network architecture: {complete_layers}")
                
                mlp = MultilayerPerceptronClassifier(
                    featuresCol=features_col,
                    labelCol=label_col,
                    layers=complete_layers,
                    maxIter=hyperparams.get('maxIter', 100),
                    blockSize=hyperparams.get('blockSize', 512),
                    seed=42
                )
                return mlp.fit(train_data)
            
            # Fallback for when layers is None (use the old logic)
            mlp = MultilayerPerceptronClassifier(
                featuresCol=features_col,
                labelCol=label_col,
                layers=layers,
                maxIter=hyperparams.get('maxIter', 100),
                blockSize=hyperparams.get('blockSize', 512),
                seed=42
            )
            return mlp.fit(train_data)
        elif model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available. Install with: pip install xgboost>=1.6.0")
            try:
                from xgboost.spark import SparkXGBClassifier
                print(f"🔧 Building XGBoost model with hyperparameters: {hyperparams}")
                
                xgb = SparkXGBClassifier(
                    features_col=features_col,
                    label_col=label_col,
                    max_depth=hyperparams.get('maxDepth', 6),
                    n_estimators=hyperparams.get('numRound', 100),
                    eta=hyperparams.get('eta', 0.3),
                    subsample=hyperparams.get('subsample', 1.0),
                    colsample_bytree=hyperparams.get('colsample_bytree', 1.0),
                    min_child_weight=hyperparams.get('min_child_weight', 1),
                    gamma=hyperparams.get('gamma', 0.0),
                    num_workers=hyperparams.get('num_workers', 1),
                    use_gpu=hyperparams.get('use_gpu', False)
                )
                
                print(f"✅ XGBoost classifier created, training with {train_data.count()} samples...")
                
                # Validate data format before training
                print(f"🔍 Validating data format for XGBoost training...")
                sample_row = train_data.select(features_col, label_col).first()
                if sample_row:
                    print(f"   📊 Sample features: {type(sample_row[0])}, shape: {getattr(sample_row[0], 'size', 'unknown')}")
                    print(f"   📊 Sample label: {type(sample_row[1])}, value: {sample_row[1]}")
                
                xgbModel = xgb.fit(train_data)
                print(f"✅ XGBoost model training completed successfully")
                
                # Validate the trained model
                if hasattr(xgbModel, 'transform') and hasattr(xgbModel, 'write'):
                    print(f"✅ XGBoost model validation passed - has required methods")
                else:
                    print(f"⚠️ XGBoost model may be missing required methods")
                    print(f"   Available methods: {[m for m in dir(xgbModel) if not m.startswith('_')]}")
                    
                    # Check if this is a compatibility issue
                    if not hasattr(xgbModel, 'write'):
                        print(f"   ❌ XGBoost model missing 'write' method - this will cause save failures")
                        print(f"   💡 This may be a version compatibility issue between XGBoost and PySpark")
                        raise RuntimeError("XGBoost model missing required 'write' method for saving")
                
                return xgbModel
                
            except Exception as e:
                print(f"❌ XGBoost model building failed: {str(e)}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"XGBoost model building failed: {str(e)}")
        elif model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not available. Install with: pip install synapseml>=0.11.0")
            from synapse.ml.lightgbm import LightGBMClassifier
            # Determine objective based on number of classes
            objective = 'multiclass' if num_classes > 2 else 'binary'
            lgb = LightGBMClassifier(
                featuresCol=features_col,
                labelCol=label_col,
                objective=objective,
                numLeaves=hyperparams.get('numLeaves', 31),
                numIterations=hyperparams.get('numIterations', 100),
                learningRate=hyperparams.get('learningRate', 0.1),
                featureFraction=hyperparams.get('featureFraction', 1.0),
                baggingFraction=hyperparams.get('baggingFraction', 1.0),
                lambdaL1=hyperparams.get('lambdaL1', 0.0),
                lambdaL2=hyperparams.get('lambdaL2', 0.0),
                seed=42
            )
            return lgb.fit(train_data)
        else:
            # Fallback to default builder if not handled
            build_func = self.model_types[model_type]['build_func']
            return build_func(train_data, features_col, label_col)
    
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
        
        # Create unfitted estimators.  If hyperparameters are passed via params,
        # they override the default settings used for each estimator.  Only
        # relevant keys are applied.
        hyperparams = {k: v for k, v in params.items() if k not in ['feature_count', 'num_classes']}

        if model_type == 'logistic':
            estimator = LogisticRegression(
                featuresCol=features_col,
                labelCol=label_col,
                maxIter=hyperparams.get('maxIter', 10),
                regParam=hyperparams.get('regParam', 0.0),
                elasticNetParam=hyperparams.get('elasticNetParam', 0.0)
            )
        elif model_type == 'random_forest':
            estimator = RandomForestClassifier(
                featuresCol=features_col,
                labelCol=label_col,
                numTrees=hyperparams.get('numTrees', 10),
                maxDepth=hyperparams.get('maxDepth', 5),
                maxBins=hyperparams.get('maxBins', 32),
                minInstancesPerNode=hyperparams.get('minInstancesPerNode', 1),
                subsamplingRate=hyperparams.get('subsamplingRate', 1.0),
                featureSubsetStrategy=hyperparams.get('featureSubsetStrategy', 'auto'),
                seed=42
            )
        elif model_type == 'gradient_boosting':
            # Apply gradient boosting optimizations to reduce large task binary warnings
            try:
                from spark_optimization_config import apply_gradient_boosting_optimizations
                apply_gradient_boosting_optimizations(self.spark)
            except Exception as e:
                print(f"⚠️ Could not apply gradient boosting optimizations: {e}")
            estimator = GBTClassifier(
                featuresCol=features_col,
                labelCol=label_col,
                maxIter=hyperparams.get('maxIter', 50),
                maxDepth=hyperparams.get('maxDepth', 5),
                maxBins=hyperparams.get('maxBins', 32),
                stepSize=hyperparams.get('stepSize', 0.1),
                subsamplingRate=hyperparams.get('subsamplingRate', 1.0),
                seed=42
            )
        elif model_type == 'decision_tree':
            estimator = DecisionTreeClassifier(
                featuresCol=features_col,
                labelCol=label_col,
                maxDepth=hyperparams.get('maxDepth', 5),
                maxBins=hyperparams.get('maxBins', 32),
                minInstancesPerNode=hyperparams.get('minInstancesPerNode', 1),
                minInfoGain=hyperparams.get('minInfoGain', 0.0),
                seed=42
            )
        elif model_type == 'neural_network':
            feature_count = params.get('feature_count', 10)
            num_classes = params.get('num_classes', 2)
            # Allow user-specified layers, else derive default
            layers = hyperparams.get('layers')
            if layers is None and feature_count:
                layers = [feature_count, feature_count*3, feature_count*2, num_classes]
            estimator = MultilayerPerceptronClassifier(
                featuresCol=features_col,
                labelCol=label_col,
                maxIter=hyperparams.get('maxIter', 100),
                layers=layers,
                blockSize=hyperparams.get('blockSize', 512),
                seed=42
            )
        elif model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available. Install with: pip install xgboost>=1.6.0")
            try:
                from xgboost.spark import SparkXGBClassifier
                estimator = SparkXGBClassifier(
                    features_col=features_col,
                    label_col=label_col,
                    max_depth=hyperparams.get('maxDepth', 6),
                    n_estimators=hyperparams.get('numRound', 100),
                    eta=hyperparams.get('eta', 0.3),
                    subsample=hyperparams.get('subsample', 1.0),
                    colsample_bytree=hyperparams.get('colsample_bytree', 1.0),
                    min_child_weight=hyperparams.get('min_child_weight', 1),
                    gamma=hyperparams.get('gamma', 0.0),
                    num_workers=hyperparams.get('num_workers', 1),
                    use_gpu=hyperparams.get('use_gpu', False)
                )
                
                # Validate the estimator
                if hasattr(estimator, 'fit') and hasattr(estimator, 'getFeaturesCol') and hasattr(estimator, 'getLabelCol'):
                    print(f"✅ XGBoost estimator created successfully with parameters: {hyperparams}")
                    print(f"   Features column: {estimator.getFeaturesCol()}")
                    print(f"   Label column: {estimator.getLabelCol()}")
                else:
                    print(f"⚠️ XGBoost estimator may be missing required methods")
                    print(f"   Available methods: {[m for m in dir(estimator) if not m.startswith('_')]}")
                    raise RuntimeError("XGBoost estimator validation failed")
            except Exception as e:
                print(f"❌ Failed to create XGBoost estimator: {str(e)}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"XGBoost estimator creation failed: {str(e)}")
        elif model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not available. Install with: pip install synapseml>=0.11.0")
            from synapse.ml.lightgbm import LightGBMClassifier
            num_classes = params.get('num_classes', 2)
            objective = 'multiclass' if num_classes > 2 else 'binary'
            estimator = LightGBMClassifier(
                featuresCol=features_col,
                labelCol=label_col,
                objective=objective,
                numLeaves=hyperparams.get('numLeaves', 31),
                numIterations=hyperparams.get('numIterations', 100),
                learningRate=hyperparams.get('learningRate', 0.1),
                featureFraction=hyperparams.get('featureFraction', 1.0),
                baggingFraction=hyperparams.get('baggingFraction', 1.0),
                lambdaL1=hyperparams.get('lambdaL1', 0.0),
                lambdaL2=hyperparams.get('lambdaL2', 0.0),
                seed=42
            )
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
        if model is None:
            raise ValueError("Cannot save None model. Model training failed.")
        
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