"""
Configuration Manager for AutoML Pipeline

This module handles loading and managing configuration from YAML files,
with support for package detection, validation, and runtime overrides.
"""

import os
import yaml
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

class ConfigManager:
    """
    Manages configuration for the AutoML pipeline.
    
    Features:
    - Loads configuration from YAML files
    - Handles package availability detection
    - Supports runtime configuration overrides
    - Provides task-specific configuration access
    - Validates configuration values
    """
    
    def __init__(self, config_path: Optional[str] = None, environment: Optional[str] = None):
        """
        Initialize the ConfigManager.
        
        Args:
            config_path: Path to the YAML configuration file.
                        If None, looks for 'config.yaml' in current directory.
            environment: Environment to use ('development', 'staging', 'production').
                        If None, uses default_environment from config or 'development'.
        """
        if config_path is None:
            config_path = "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.environment = environment or self.config.get('default_environment', 'development')
        
        print(f"🌍 Using environment: {self.environment}")
        
        self._apply_environment_config()
        self._detect_package_availability()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                print(f"✅ Configuration loaded from {self.config_path}")
                return config
        except FileNotFoundError:
            print(f"⚠️ Configuration file {self.config_path} not found. Using default configuration.")
            return self._get_default_config()
        except yaml.YAMLError as e:
            print(f"❌ Error parsing YAML file {self.config_path}: {e}")
            print("🔄 Falling back to default configuration.")
            return self._get_default_config()
    
    def _detect_package_availability(self):
        """Detect availability of optional packages and update configuration."""
        # Check XGBoost availability
        try:
            import xgboost.spark
            xgboost_available = True
            print("✅ XGBoost detected and available")
        except ImportError:
            xgboost_available = False
            print("⚠️ XGBoost not available")
        
        # Check LightGBM availability
        try:
            import synapse.ml.lightgbm
            lightgbm_available = True
            print("✅ LightGBM (SynapseML) detected and available")
        except ImportError:
            lightgbm_available = False
            print("⚠️ LightGBM (SynapseML) not available")
        
        # Update configuration for all task types
        for task_type in ['classification', 'regression']:
            if task_type in self.config:
                models = self.config[task_type].get('models', {})
                
                # Update XGBoost availability
                if models.get('run_xgboost') == 'auto':
                    models['run_xgboost'] = xgboost_available
                
                # Update LightGBM availability
                if models.get('run_lightgbm') == 'auto':
                    models['run_lightgbm'] = lightgbm_available
    
    def _apply_environment_config(self):
        """Apply environment-specific configuration overrides."""
        if 'environments' not in self.config:
            print("⚠️ No environment configurations found in config file")
            return
        
        environments = self.config['environments']
        
        if self.environment not in environments:
            available_envs = list(environments.keys())
            print(f"⚠️ Environment '{self.environment}' not found.")
            print(f"   Available environments: {available_envs}")
            print(f"   Using default configuration")
            return
        
        env_config = environments[self.environment]
        print(f"✅ Applying {self.environment} environment configuration")
        
        # Apply environment-specific overrides
        for section_name, section_config in env_config.items():
            if section_name in self.config:
                # Merge with existing section
                self._deep_merge(self.config[section_name], section_config)
            else:
                # Add new section
                self.config[section_name] = section_config.copy()
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]):
        """Deep merge source dictionary into target dictionary."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
    
    def switch_environment(self, new_environment: str):
        """
        Switch to a different environment and reload configuration.
        
        Args:
            new_environment: New environment to switch to
        """
        print(f"🔄 Switching from {self.environment} to {new_environment}")
        
        # Reload base configuration to start fresh
        self.config = self._load_config()
        self.environment = new_environment
        
        # Apply new environment configuration
        self._apply_environment_config()
        self._detect_package_availability()
        
        print(f"✅ Successfully switched to {new_environment} environment")
    
    def get_available_environments(self) -> List[str]:
        """Get list of available environments."""
        if 'environments' not in self.config:
            return []
        return list(self.config['environments'].keys())
    
    def get_current_environment(self) -> str:
        """Get the currently active environment."""
        return self.environment
    
    def print_environment_comparison(self):
        """Print a comparison of all available environments."""
        if 'environments' not in self.config:
            print("❌ No environment configurations available")
            return
        
        environments = self.config['environments']
        
        print("\n" + "="*80)
        print("🌍 ENVIRONMENT COMPARISON")
        print("="*80)
        
        for env_name, env_config in environments.items():
            current_marker = " (CURRENT)" if env_name == self.environment else ""
            print(f"\n🎯 {env_name.upper()}{current_marker}:")
            
            # Show key differences
            if 'performance' in env_config:
                perf = env_config['performance']
                print(f"   ⚡ Performance: {perf.get('parallel_jobs', 'default')} jobs, "
                      f"{perf.get('memory_limit_gb', 'default')}GB RAM, "
                      f"{perf.get('timeout_minutes', 'default')}min timeout")
            
            if 'classification' in env_config:
                cls_config = env_config['classification']
                
                if 'models' in cls_config:
                    models = cls_config['models']
                    enabled_models = [k.replace('run_', '') for k, v in models.items() if v is True]
                    print(f"   🤖 Models: {', '.join(enabled_models) if enabled_models else 'default'}")
                
                if 'hyperparameter_tuning' in cls_config:
                    hp_config = cls_config['hyperparameter_tuning']
                    hp_enabled = hp_config.get('enable_hyperparameter_tuning', 'default')
                    hp_method = hp_config.get('optimization_method', 'default')
                    hp_trials = hp_config.get('optuna_trials', hp_config.get('random_search_trials', 'default'))
                    print(f"   🎯 Hyperparameter tuning: {hp_enabled} ({hp_method}, {hp_trials} trials)")
            
            if 'logging' in env_config:
                log_config = env_config['logging']
                log_level = log_config.get('level', 'default')
                detailed = log_config.get('detailed_metrics', 'default')
                print(f"   📋 Logging: {log_level} level, detailed={detailed}")
        
        print("="*80)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Provide default configuration as fallback."""
        return {
            'global': {
                'data_processing': {
                    'missing_value_threshold': 0.7,
                    'categorical_threshold': 10,
                    'random_seed': 42
                },
                'feature_selection': {
                    'max_features': 50,
                    'sequential_threshold': 200
                }
            },
            'classification': {
                'models': {
                    'run_logistic': True,
                    'run_random_forest': True,
                    'run_gradient_boosting': True,
                    'run_neural_network': True,
                    'run_decision_tree': True,
                    'run_xgboost': False,
                    'run_lightgbm': False
                },
                'hyperparameter_tuning': {
                    'enable_hyperparameter_tuning': False,
                    'optimization_method': 'optuna'
                }
            }
        }
    
    def get_global_config(self) -> Dict[str, Any]:
        """Get global configuration that applies to all tasks."""
        return self.config.get('global', {})
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration (alias for accessing self.config directly).
        
        Returns:
            Current configuration dictionary
        """
        return self.config
    
    def get_flexible_data_input_config(self) -> Dict[str, Any]:
        """
        Get flexible data input configuration.
        
        Returns:
            Flexible data input configuration with defaults
        """
        flexible_config = self.config.get('flexible_data_input', {})
        
        # Provide defaults if not configured
        default_flexible_config = {
            'enabled': True,
            'default_source_type': 'auto',
            'bigquery': {
                'default_options': {
                    'useAvroLogicalTypes': True,
                    'viewsEnabled': True
                },
                'authentication': {
                    'method': 'auto',
                    'service_account_path': ''
                },
                'performance': {
                    'materializationDataset': '',
                    'materializationProject': '',
                    'maxReadParallelism': 10000
                }
            },
            'file_upload': {
                'supported_formats': {
                    'csv': ['.csv'],
                    'tsv': ['.tsv', '.tab'],
                    'excel': ['.xlsx', '.xls'],
                    'json': ['.json'],
                    'parquet': ['.parquet']
                },
                'default_options': {
                    'csv': {
                        'delimiter': ',',
                        'header': True,
                        'infer_schema': True,
                        'encoding': 'utf-8'
                    },
                    'tsv': {
                        'delimiter': '\t',
                        'header': True,
                        'infer_schema': True,
                        'encoding': 'utf-8'
                    },
                    'excel': {
                        'sheet_name': 0,
                        'header': 0
                    },
                    'json': {
                        'multiline': True
                    },
                    'parquet': {}
                },
                'save_uploaded_files': True,
                'timestamp_uploaded_files': True,
                'max_file_size_mb': 500
            },
            'existing_files': {
                'search_directories': ['.', './'],
                'known_datasets': ['bank.csv', 'IRIS.csv', 'bank', 'iris'],
                'case_sensitive': False,
                'search_extensions': ['.csv', '.xlsx', '.xls', '.json', '.parquet', '.tsv', '.tab']
            },
            'preview': {
                'enabled': True,
                'max_rows': 5,
                'show_schema': True,
                'show_statistics': True
            },
            'validation': {
                'enabled': True,
                'check_null_values': True,
                'check_data_types': True,
                'check_column_names': True,
                'max_validation_rows': 10000
            },
            'error_handling': {
                'retry_attempts': 3,
                'fallback_to_basic': True,
                'detailed_error_messages': True
            },
            'caching': {
                'enabled': False,
                'cache_directory': 'temp/data_cache',
                'cache_expiry_hours': 24,
                'max_cache_size_gb': 2
            }
        }
        
        # Merge with provided config (deep merge)
        merged_config = self._deep_merge_dicts(default_flexible_config, flexible_config)
        return merged_config
    
    def get_bigquery_config(self) -> Dict[str, Any]:
        """
        Get BigQuery-specific configuration.
        
        Returns:
            BigQuery configuration dictionary
        """
        flexible_config = self.get_flexible_data_input_config()
        return flexible_config.get('bigquery', {})
    
    def get_file_upload_config(self) -> Dict[str, Any]:
        """
        Get file upload configuration.
        
        Returns:
            File upload configuration dictionary
        """
        flexible_config = self.get_flexible_data_input_config()
        return flexible_config.get('file_upload', {})
    
    def get_existing_files_config(self) -> Dict[str, Any]:
        """
        Get existing files configuration.
        
        Returns:
            Existing files configuration dictionary
        """
        flexible_config = self.get_flexible_data_input_config()
        return flexible_config.get('existing_files', {})
    
    def is_flexible_data_input_enabled(self) -> bool:
        """
        Check if flexible data input system is enabled.
        
        Returns:
            True if flexible data input is enabled
        """
        flexible_config = self.get_flexible_data_input_config()
        return flexible_config.get('enabled', True)
    
    def get_default_source_type(self) -> str:
        """
        Get the default source type for auto-detection.
        
        Returns:
            Default source type ('auto', 'existing', 'upload', 'bigquery')
        """
        flexible_config = self.get_flexible_data_input_config()
        return flexible_config.get('default_source_type', 'auto')
    
    def get_supported_file_formats(self) -> Dict[str, List[str]]:
        """
        Get supported file formats for uploads.
        
        Returns:
            Dictionary mapping format names to file extensions
        """
        file_config = self.get_file_upload_config()
        return file_config.get('supported_formats', {
            'csv': ['.csv'],
            'tsv': ['.tsv', '.tab'],
            'excel': ['.xlsx', '.xls'],
            'json': ['.json'],
            'parquet': ['.parquet']
        })
    
    def get_format_default_options(self, format_name: str) -> Dict[str, Any]:
        """
        Get default options for a specific file format.
        
        Args:
            format_name: Name of the format ('csv', 'tsv', 'excel', 'json', 'parquet')
            
        Returns:
            Default options dictionary for the format
        """
        file_config = self.get_file_upload_config()
        default_options = file_config.get('default_options', {})
        return default_options.get(format_name, {})
    
    def _deep_merge_dicts(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries, with dict2 values taking precedence.
        
        Args:
            dict1: Base dictionary
            dict2: Dictionary to merge (takes precedence)
            
        Returns:
            Merged dictionary
        """
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dicts(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def update_flexible_data_input_config(self, updates: Dict[str, Any]) -> None:
        """
        Update flexible data input configuration.
        
        Args:
            updates: Dictionary of configuration updates
        """
        if 'flexible_data_input' not in self.config:
            self.config['flexible_data_input'] = {}
        
        # Deep merge the updates
        self.config['flexible_data_input'] = self._deep_merge_dicts(
            self.config['flexible_data_input'], 
            updates
        )
    
    def validate_flexible_data_input_config(self) -> List[str]:
        """
        Validate flexible data input configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        try:
            flexible_config = self.get_flexible_data_input_config()
            
            # Check if enabled setting is boolean
            enabled = flexible_config.get('enabled')
            if not isinstance(enabled, bool):
                errors.append("flexible_data_input.enabled must be a boolean")
            
            # Check default source type
            default_source = flexible_config.get('default_source_type')
            valid_sources = ['auto', 'existing', 'upload', 'bigquery']
            if default_source not in valid_sources:
                errors.append(f"flexible_data_input.default_source_type must be one of {valid_sources}")
            
            # Validate BigQuery config
            bigquery_config = flexible_config.get('bigquery', {})
            if 'authentication' in bigquery_config:
                auth_method = bigquery_config['authentication'].get('method')
                valid_auth_methods = ['auto', 'service_account', 'gcloud']
                if auth_method not in valid_auth_methods:
                    errors.append(f"BigQuery authentication method must be one of {valid_auth_methods}")
            
            # Validate file upload config
            file_config = flexible_config.get('file_upload', {})
            max_size = file_config.get('max_file_size_mb')
            if max_size is not None and (not isinstance(max_size, (int, float)) or max_size <= 0):
                errors.append("file_upload.max_file_size_mb must be a positive number")
            
            # Validate preview config
            preview_config = flexible_config.get('preview', {})
            max_rows = preview_config.get('max_rows')
            if max_rows is not None and (not isinstance(max_rows, int) or max_rows <= 0):
                errors.append("preview.max_rows must be a positive integer")
                
        except Exception as e:
            errors.append(f"Error validating flexible data input config: {e}")
        
        return errors

    def get_task_config(self, task_type: str) -> Dict[str, Any]:
        """
        Get configuration for a specific task type.
        
        Args:
            task_type: Type of task ('classification', 'regression', 'clustering')
            
        Returns:
            Task-specific configuration dictionary
        """
        if task_type not in self.config:
            print(f"⚠️ Configuration for task '{task_type}' not found. Using defaults.")
            return {}
        
        return self.config[task_type]
    
    def get_flat_config(self, task_type: str, include_global: bool = True) -> Dict[str, Any]:
        """
        Get a flattened configuration suitable for backwards compatibility.
        
        Args:
            task_type: Type of task ('classification', 'regression', 'clustering')
            include_global: Whether to include global settings
            
        Returns:
            Flattened configuration dictionary
        """
        flat_config = {}
        
        # Add global settings if requested
        if include_global:
            global_config = self.get_global_config()
            flat_config.update(self._flatten_dict(global_config))
        
        # Add task-specific settings
        task_config = self.get_task_config(task_type)
        flat_config.update(self._flatten_dict(task_config))
        
        return flat_config
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten a nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def override_config(self, overrides: Dict[str, Any], task_type: Optional[str] = None):
        """
        Override configuration values at runtime.
        
        Args:
            overrides: Dictionary of configuration overrides
            task_type: If specified, apply overrides to this task type only
        """
        if task_type:
            # Apply overrides to specific task
            if task_type not in self.config:
                self.config[task_type] = {}
            self._merge_dict(self.config[task_type], overrides)
        else:
            # Apply overrides globally
            self._merge_dict(self.config, overrides)
        
        print(f"🔄 Configuration overridden with {len(overrides)} values")
    
    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration with new values (alias for override_config for backward compatibility).
        
        Args:
            updates: Dictionary of configuration updates
        """
        self.override_config(updates)
    
    def _merge_dict(self, target: Dict[str, Any], source: Dict[str, Any]):
        """Recursively merge source dictionary into target."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_dict(target[key], value)
            else:
                target[key] = value
    
    def validate_config(self, task_type: str) -> bool:
        """
        Validate configuration for a specific task type.
        
        Args:
            task_type: Type of task to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        task_config = self.get_task_config(task_type)
        
        if not task_config:
            print(f"❌ No configuration found for task type: {task_type}")
            return False
        
        # Validate required sections
        required_sections = ['models']
        for section in required_sections:
            if section not in task_config:
                print(f"❌ Missing required section '{section}' in {task_type} configuration")
                return False
        
        # Validate at least one model is enabled
        models = task_config.get('models', {})
        enabled_models = [k for k, v in models.items() if v is True]
        
        if not enabled_models:
            print(f"❌ No models enabled in {task_type} configuration")
            return False
        
        print(f"✅ Configuration validation passed for {task_type}")
        return True
    
    def save_config(self, output_path: Optional[str] = None):
        """
        Save current configuration to YAML file.
        
        Args:
            output_path: Path to save configuration. If None, overwrites original file.
        """
        if output_path is None:
            output_path = str(self.config_path)
        
        try:
            with open(output_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False, sort_keys=False)
            print(f"💾 Configuration saved to {output_path}")
        except Exception as e:
            print(f"❌ Error saving configuration: {e}")
    
    def print_config_summary(self, task_type: Optional[str] = None):
        """
        Print a summary of the current configuration.
        
        Args:
            task_type: If specified, show summary for this task type only
        """
        print("\n" + "="*60)
        print("📋 CONFIGURATION SUMMARY")
        print("="*60)
        
        if task_type:
            # Show specific task configuration
            task_config = self.get_task_config(task_type)
            if task_config:
                print(f"\n🎯 {task_type.upper()} Configuration:")
                self._print_config_section(task_config)
            else:
                print(f"❌ No configuration found for {task_type}")
        else:
            # Show all configurations
            print(f"\n🌐 Global Configuration:")
            self._print_config_section(self.get_global_config())
            
            for task in ['classification', 'regression', 'clustering']:
                if task in self.config:
                    print(f"\n🎯 {task.upper()} Configuration:")
                    self._print_config_section(self.config[task])
        
        print("="*60)
    
    def _print_config_section(self, config: Dict[str, Any], indent: int = 0):
        """Print a configuration section with proper indentation."""
        spaces = "  " * indent
        for key, value in config.items():
            if isinstance(value, dict):
                print(f"{spaces}📁 {key}:")
                self._print_config_section(value, indent + 1)
            else:
                print(f"{spaces}⚙️  {key}: {value}")
    
    def get_model_selection_criteria(self, task_type: str) -> str:
        """Get model selection criteria for a specific task type."""
        task_config = self.get_task_config(task_type)
        return task_config.get('evaluation', {}).get('model_selection_criteria', 'accuracy')
    
    def get_enabled_models(self, task_type: str) -> Dict[str, bool]:
        """Get dictionary of enabled models for a specific task type."""
        task_config = self.get_task_config(task_type)
        return task_config.get('models', {})
    
    def is_hyperparameter_tuning_enabled(self, task_type: str) -> bool:
        """Check if hyperparameter tuning is enabled for a task type."""
        task_config = self.get_task_config(task_type)
        return task_config.get('hyperparameter_tuning', {}).get('enable_hyperparameter_tuning', False)
    
    def apply_preset(self, preset: str):
        """
        Apply preset configuration for quick or comprehensive training.
        
        Args:
            preset: Either 'quick', 'comprehensive', or empty string for no preset
        """
        # Handle empty preset (no preset selected)
        if not preset or preset.strip() == '':
            print("   📝 No preset selected - using configuration from YAML file")
            return
        
        valid_presets = ['quick', 'comprehensive']
        if preset not in valid_presets:
            raise ValueError(f"Invalid preset '{preset}'. Must be one of: {valid_presets} or empty string")
        
        if preset == 'quick':
            print("   ⚡ Quick preset: Limited models, no CV, no hyperparameter tuning")
            quick_overrides = {
                'classification': {
                    'models': {
                        'run_logistic': True,
                        'run_random_forest': True,
                        'run_gradient_boosting': False,
                        'run_neural_network': False,
                        'run_decision_tree': False,
                        'run_xgboost': False,
                        'run_lightgbm': False
                    },
                    'hyperparameter_tuning': {
                        'enable_hyperparameter_tuning': False,
                        'optimization_method': 'random_search'
                    },
                    'cross_validation': {
                        'use_cross_validation': False,
                        'cv_folds': 5
                    }
                },
                'regression': {
                    'models': {
                        'run_linear_regression': True,
                        'run_random_forest': True,
                        'run_gradient_boosting': False,
                        'run_decision_tree': False,
                        'run_xgboost': False,
                        'run_lightgbm': False
                    },
                    'hyperparameter_tuning': {
                        'enable_hyperparameter_tuning': False,
                        'optimization_method': 'random_search'
                    },
                    'cross_validation': {
                        'use_cross_validation': False,
                        'cv_folds': 5
                    }
                },
                'clustering': {
                    'models': {
                        'run_kmeans': True,
                        'run_bisecting_kmeans': False,
                        'run_dbscan': False,
                        'run_gaussian_mixture': False
                    },
                    'hyperparameter_tuning': {
                        'enable_hyperparameter_tuning': False,
                        'optimization_method': 'random_search'
                    },
                    'evaluation': {
                        'evaluation_method': 'silhouette'
                    }
                }
            }
            self.override_config(quick_overrides)
            
        elif preset == 'comprehensive':
            print("   🔬 Comprehensive preset: All models enabled, full hyperparameter tuning")
            comprehensive_overrides = {
                'classification': {
                    'models': {
                        'run_logistic': True,
                        'run_random_forest': True,
                        'run_gradient_boosting': True,
                        'run_neural_network': True,
                        'run_decision_tree': True,
                        'run_xgboost': True,  # Will be auto-disabled if not available
                        'run_lightgbm': True  # Will be auto-disabled if not available
                    },
                    'hyperparameter_tuning': {
                        'enable_hyperparameter_tuning': True,
                        'optimization_method': 'optuna',
                        'optuna_trials': 100,
                        'optuna_timeout': 3600
                    },
                    'cross_validation': {
                        'use_cross_validation': 'auto',  # Let existing logic decide
                        'cv_folds': 5
                    }
                },
                'regression': {
                    'models': {
                        'run_linear_regression': True,
                        'run_random_forest': True,
                        'run_gradient_boosting': True,
                        'run_decision_tree': True,
                        'run_xgboost': True,  # Will be auto-disabled if not available
                        'run_lightgbm': True  # Will be auto-disabled if not available
                    },
                    'hyperparameter_tuning': {
                        'enable_hyperparameter_tuning': True,
                        'optimization_method': 'optuna',
                        'optuna_trials': 100,
                        'optuna_timeout': 3600
                    },
                    'cross_validation': {
                        'use_cross_validation': 'auto',  # Let existing logic decide
                        'cv_folds': 5
                    }
                },
                'clustering': {
                    'models': {
                        'run_kmeans': True,
                        'run_bisecting_kmeans': True,
                        'run_dbscan': True,
                        'run_gaussian_mixture': True
                    },
                    'hyperparameter_tuning': {
                        'enable_hyperparameter_tuning': True,
                        'optimization_method': 'optuna',
                        'optuna_trials': 50,
                        'optuna_timeout': 1800
                    },
                    'evaluation': {
                        'evaluation_method': 'silhouette'
                    }
                }
            }
            self.override_config(comprehensive_overrides)


# Convenience function to get a configured ConfigManager instance
def get_config_manager(config_path: Optional[str] = None, environment: Optional[str] = None) -> ConfigManager:
    """
    Get a ConfigManager instance.
    
    Args:
        config_path: Path to configuration file. If None, uses default.
        environment: Environment to use ('development', 'staging', 'production').
        
    Returns:
        Configured ConfigManager instance
    """
    return ConfigManager(config_path, environment)


# Example usage and testing
if __name__ == "__main__":
    # Create config manager
    config_manager = ConfigManager()
    
    # Print configuration summary
    config_manager.print_config_summary()
    
    # Test task-specific configuration
    classification_config = config_manager.get_task_config('classification')
    print(f"\nClassification models: {config_manager.get_enabled_models('classification')}")
    
    # Test validation
    is_valid = config_manager.validate_config('classification')
    print(f"Configuration valid: {is_valid}") 