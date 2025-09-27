# CROP YIELD PREDICTION WITH CNN
# ===============================

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from pathlib import Path
import cv2
import json
from datetime import datetime

class CropYieldPredictor:
    """Complete Crop Yield Prediction System"""
    
    def __init__(self, project_dir="./"):
        self.project_dir = Path(project_dir)
        self.data_dir = self.project_dir / "data"
        self.models_dir = self.project_dir / "models" 
        self.results_dir = self.project_dir / "results"
        
        # Create directories
        for dir_path in [self.data_dir, self.models_dir, self.results_dir]:
            dir_path.mkdir(exist_ok=True)
            
        self.model = None
        self.history = None
        
    def setup_logging(self):
        """Setup logging for the training process"""
        import logging
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.results_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def create_synthetic_dataset(self, num_samples=1000):
        """Create comprehensive synthetic agricultural dataset"""
        logger = self.setup_logging()
        logger.info(f"Creating synthetic dataset with {num_samples} samples...")
        
        # Create images directory
        images_dir = self.data_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Generate realistic agricultural data
        np.random.seed(42)
        data = []
        
        logger.info("Generating synthetic field parameters...")
        
        for i in range(num_samples):
            # Agricultural parameters with realistic correlations
            field_id = f"F{i//25:03d}"  # 25 plots per field
            plot_id = f"P{i%25:02d}"
            
            # Environmental factors
            vegetation_index = np.clip(np.random.normal(0.7, 0.15), 0.2, 1.0)
            soil_quality = np.clip(np.random.normal(0.65, 0.18), 0.1, 1.0)
            weather_score = np.clip(np.random.normal(0.75, 0.12), 0.3, 1.0)
            water_availability = np.clip(np.random.normal(0.7, 0.15), 0.2, 1.0)
            
            # Growth stage
            growth_stages = ['vegetative', 'flowering', 'grain_filling', 'maturity']
            growth_stage = np.random.choice(growth_stages)
            
            # Stage multiplier
            stage_multipliers = {
                'vegetative': 0.6, 'flowering': 0.8, 
                'grain_filling': 1.0, 'maturity': 0.9
            }
            
            # Calculate realistic yield with multiple factors
            base_yield = 5.5  # tons per hectare
            yield_value = (base_yield * vegetation_index * soil_quality * 
                          weather_score * water_availability * 
                          stage_multipliers[growth_stage] + 
                          np.random.normal(0, 0.25))
            
            yield_value = np.clip(yield_value, 1.0, 8.5)
            
            # Create and save synthetic image
            img_filename = f"{field_id}_{plot_id}_{growth_stage}.jpg"
            img = self.generate_field_image(
                vegetation_index, soil_quality, weather_score, 
                growth_stage, yield_value, i
            )
            
            cv2.imwrite(str(images_dir / img_filename), img)
            
            # Store data
            data.append({
                'image_filename': img_filename,
                'field_id': field_id,
                'plot_id': plot_id,
                'yield_tons_per_hectare': round(yield_value, 2),
                'vegetation_index': round(vegetation_index, 3),
                'soil_quality': round(soil_quality, 3),
                'weather_score': round(weather_score, 3),
                'water_availability': round(water_availability, 3),
                'growth_stage': growth_stage,
                'latitude': round(40.7128 + np.random.normal(0, 0.1), 4),
                'longitude': round(-74.0060 + np.random.normal(0, 0.1), 4),
                'planting_date': '2024-04-15',
                'image_date': '2024-07-15'
            })
            
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{num_samples} samples")
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        csv_path = self.data_dir / "crop_yield_dataset.csv"
        df.to_csv(csv_path, index=False)
        
        # Save dataset statistics
        stats = {
            'total_samples': len(df),
            'yield_range': [float(df['yield_tons_per_hectare'].min()), 
                           float(df['yield_tons_per_hectare'].max())],
            'mean_yield': float(df['yield_tons_per_hectare'].mean()),
            'std_yield': float(df['yield_tons_per_hectare'].std()),
            'growth_stages': df['growth_stage'].value_counts().to_dict(),
            'creation_date': datetime.now().isoformat()
        }
        
        with open(self.data_dir / "dataset_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Dataset created successfully!")
        logger.info(f"  Location: {csv_path}")
        logger.info(f"  Samples: {len(df)}")
        logger.info(f"  Yield range: {stats['yield_range'][0]:.1f} - {stats['yield_range'][1]:.1f} tons/ha")
        logger.info(f"  Mean yield: {stats['mean_yield']:.2f} ± {stats['std_yield']:.2f} tons/ha")
        
        return df
    
    def generate_field_image(self, veg_index, soil_quality, weather, 
                           growth_stage, yield_value, seed):
        """Generate realistic synthetic field images"""
        np.random.seed(seed)
        
        height, width = 224, 224
        
        # Base colors based on parameters
        green_base = int(30 + veg_index * 180)
        brown_base = int(80 + soil_quality * 100)
        
        # Initialize image
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create base soil/vegetation texture
        for y in range(height):
            for x in range(width):
                # Vegetation coverage based on yield
                if np.random.random() < veg_index * 0.8:
                    # Vegetation pixel
                    if growth_stage == 'vegetative':
                        img[y, x] = [0, green_base + np.random.randint(-20, 20), 0]
                    elif growth_stage == 'flowering':
                        img[y, x] = [np.random.randint(0, 30), green_base, np.random.randint(0, 20)]
                    elif growth_stage == 'grain_filling':
                        img[y, x] = [green_base//2, green_base, np.random.randint(0, 15)]
                    else:  # maturity
                        img[y, x] = [brown_base, green_base//2, 0]
                else:
                    # Soil pixel
                    soil_variation = np.random.randint(-30, 30)
                    img[y, x] = [brown_base + soil_variation, 
                               (brown_base + soil_variation)//2, 0]
        
        # Add crop rows pattern
        row_spacing = 15
        for row in range(0, height, row_spacing):
            if row + 2 < height:
                img[row:row+2, :] = img[row:row+2, :] * 0.7  # Darker soil between rows
        
        # Add weather effects
        if weather < 0.5:  # Poor weather - add brown/yellow tinge
            img[:, :, 0] = np.clip(img[:, :, 0] * 1.3, 0, 255)
            img[:, :, 1] = np.clip(img[:, :, 1] * 0.8, 0, 255)
        
        # High yield areas - denser vegetation spots
        if yield_value > 6.0:
            for _ in range(int((yield_value - 4) * 20)):
                center_x = np.random.randint(5, width-5)
                center_y = np.random.randint(5, height-5)
                radius = np.random.randint(2, 5)
                cv2.circle(img, (center_x, center_y), radius, 
                          (0, min(255, green_base + 40), 0), -1)
        
        # Add some realistic noise
        noise = np.random.randint(-10, 10, (height, width, 3))
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Apply slight blur for realism
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        return img
    
    def build_cnn_model(self):
        """Build optimized CNN for yield prediction"""
        logger = self.setup_logging()
        logger.info("Building CNN model architecture...")
        
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(224, 224, 3)),
            
            # First convolutional block
            layers.Conv2D(32, (7, 7), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block  
            layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Global pooling and dense layers
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(), 
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer for regression
            layers.Dense(1, activation='linear')
        ])
        
        # Compile with advanced optimizer
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
            loss='huber',  # More robust to outliers than MSE
            metrics=['mae', 'mse']
        )
        
        self.model = model
        
        logger.info("Model architecture built successfully!")
        logger.info(f"Total parameters: {model.count_params():,}")
        
        return model
    
    def prepare_data(self, df, test_size=0.15, val_size=0.15):
        """Prepare stratified train/validation/test splits"""
        logger = self.setup_logging()
        logger.info("Preparing data splits...")
        
        # Create yield bins for stratified splitting
        df['yield_bin'] = pd.cut(df['yield_tons_per_hectare'], 
                               bins=5, labels=['low', 'med_low', 'med', 'med_high', 'high'])
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42, 
            stratify=df['yield_bin']
        )
        
        # Second split: separate validation from training
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size/(1-test_size), random_state=42,
            stratify=train_val_df['yield_bin']
        )
        
        # Remove temporary column
        for split_df in [train_df, val_df, test_df]:
            split_df.drop('yield_bin', axis=1, inplace=True)
        
        logger.info(f"Data splits created:")
        logger.info(f"  Training: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
        logger.info(f"  Validation: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
        logger.info(f"  Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def create_data_generator(self, df, batch_size=16, shuffle=True, augment=False):
        """Create efficient data generator with optional augmentation"""
        
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        # Setup augmentation if needed
        if augment:
            datagen = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1, 
                zoom_range=0.1,
                horizontal_flip=True,
                brightness_range=[0.8, 1.2],
                fill_mode='nearest'
            )
        
        def generator():
            while True:
                if shuffle:
                    df_shuffled = df.sample(frac=1, random_state=None).reset_index(drop=True)
                else:
                    df_shuffled = df
                
                for start_idx in range(0, len(df_shuffled), batch_size):
                    batch_df = df_shuffled.iloc[start_idx:start_idx + batch_size]
                    
                    batch_images = []
                    batch_yields = []
                    
                    for _, row in batch_df.iterrows():
                        try:
                            # Load and preprocess image
                            img_path = self.data_dir / "images" / row['image_filename']
                            img = cv2.imread(str(img_path))
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = img.astype(np.float32) / 255.0
                            
                            # Apply augmentation if enabled
                            if augment:
                                img = datagen.random_transform(img)
                            
                            batch_images.append(img)
                            batch_yields.append(row['yield_tons_per_hectare'])
                            
                        except Exception as e:
                            print(f"Error loading image {row['image_filename']}: {e}")
                            continue
                    
                    if len(batch_images) > 0:
                        yield np.array(batch_images), np.array(batch_yields)
        
        return generator
    
    def train_model(self, train_df, val_df, epochs=40, batch_size=16):
        """Train the model with comprehensive monitoring"""
        logger = self.setup_logging()
        logger.info(f"Starting training for {epochs} epochs...")
        
        # Create data generators
        train_gen = self.create_data_generator(
            train_df, batch_size=batch_size, shuffle=True, augment=True
        )()
        
        val_gen = self.create_data_generator(
            val_df, batch_size=batch_size, shuffle=False, augment=False
        )()
        
        # Calculate steps
        steps_per_epoch = max(1, len(train_df) // batch_size)
        validation_steps = max(1, len(val_df) // batch_size)
        
        # Advanced callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=12, 
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=6,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                str(self.models_dir / 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            callbacks.CSVLogger(
                str(self.results_dir / 'training_history.csv')
            )
        ]
        
        # Train model
        logger.info("Training started...")
        history = self.model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.history = history
        logger.info("Training completed!")
        
        return history
    
    def evaluate_model(self, test_df, batch_size=16):
        """Comprehensive model evaluation"""
        logger = self.setup_logging()
        logger.info("Evaluating model on test set...")
        
        # Load test data
        test_images = []
        test_yields = []
        
        for _, row in test_df.iterrows():
            try:
                img_path = self.data_dir / "images" / row['image_filename']
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                
                test_images.append(img)
                test_yields.append(row['yield_tons_per_hectare'])
            except Exception as e:
                logger.warning(f"Error loading test image: {e}")
                continue
        
        test_images = np.array(test_images)
        test_yields = np.array(test_yields)
        
        # Get predictions
        logger.info("Making predictions...")
        predictions = self.model.predict(test_images, batch_size=batch_size, verbose=1)
        predictions = predictions.flatten()
        
        # Calculate comprehensive metrics
        mse = mean_squared_error(test_yields, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_yields, predictions) 
        r2 = r2_score(test_yields, predictions)
        
        # Calculate additional metrics
        mape = np.mean(np.abs((test_yields - predictions) / test_yields)) * 100
        
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2_score': float(r2),
            'mape': float(mape),
            'mean_yield_actual': float(test_yields.mean()),
            'mean_yield_predicted': float(predictions.mean()),
            'std_yield_actual': float(test_yields.std()),
            'std_yield_predicted': float(predictions.std())
        }
        
        # Save metrics
        with open(self.results_dir / 'evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("Model Evaluation Results:")
        logger.info(f"  RMSE: {rmse:.3f} tons/ha")
        logger.info(f"  MAE: {mae:.3f} tons/ha") 
        logger.info(f"  R² Score: {r2:.3f}")
        logger.info(f"  MAPE: {mape:.1f}%")
        
        # Create visualizations
        self.create_evaluation_plots(test_yields, predictions)
        
        return metrics, predictions
    
    def create_evaluation_plots(self, y_true, y_pred):
        """Create comprehensive evaluation plots"""
        # Set style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Training history
        if self.history:
            axes[0, 0].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
            axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
            axes[0, 0].set_title('Training History - Loss', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss (Huber)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # MAE history
            axes[0, 1].plot(self.history.history['mae'], label='Training MAE', linewidth=2)
            axes[0, 1].plot(self.history.history['val_mae'], label='Validation MAE', linewidth=2)
            axes[0, 1].set_title('Training History - MAE', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Mean Absolute Error')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Predictions vs Actual
        axes[1, 0].scatter(y_true, y_pred, alpha=0.6, s=30)
        axes[1, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                       'r--', lw=2, label='Perfect Prediction')
        axes[1, 0].set_xlabel('Actual Yield (tons/ha)')
        axes[1, 0].set_ylabel('Predicted Yield (tons/ha)')
        axes[1, 0].set_title('Predicted vs Actual Yield', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add R² to the plot
        r2 = r2_score(y_true, y_pred)
        axes[1, 0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[1, 0].transAxes, 
                       fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        # Residual plot
        residuals = y_pred - y_true
        axes[1, 1].scatter(y_pred, residuals, alpha=0.6, s=30)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Predicted Yield (tons/ha)')
        axes[1, 1].set_ylabel('Residuals (Predicted - Actual)')
        axes[1, 1].set_title('Residual Plot', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'evaluation_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Additional distribution plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(y_true, bins=30, alpha=0.7, label='Actual Yield', color='blue')
        plt.hist(y_pred, bins=30, alpha=0.7, label='Predicted Yield', color='red')
        plt.xlabel('Yield (tons/ha)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Actual vs Predicted Yields')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(residuals, bins=30, alpha=0.7, color='green')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Distribution of Residuals')
        plt.axvline(x=0, color='red', linestyle='--')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'distribution_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filename="final_yield_prediction_model.h5"):
        """Save the trained model with metadata"""
        model_path = self.models_dir / filename
        self.model.save(str(model_path))
        
        # Save model metadata
        metadata = {
            'model_filename': filename,
            'creation_date': datetime.now().isoformat(),
            'model_parameters': self.model.count_params(),
            'input_shape': [224, 224, 3],
            'output_shape': [1],
            'architecture': 'CNN_Regression',
            'optimizer': 'Adam',
            'loss_function': 'Huber'
        }
        
        with open(self.models_dir / f'{filename.replace(".h5", "_metadata.json")}', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Model saved: {model_path}")
        return str(model_path)

    def predict_single_image(self, image_path):
        """Predict yield for a single image"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Load and preprocess image
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        prediction = self.model.predict(img, verbose=0)
        return float(prediction[0][0])

# Main execution function
def run_complete_pipeline():
    """Execute the complete crop yield prediction pipeline"""
    print("🌾 CROP YIELD PREDICTION - COMPLETE PIPELINE")
    print("=" * 55)
    
    # Initialize predictor
    predictor = CropYieldPredictor()
    
    try:
        # Step 1: Create synthetic dataset
        print("\n🔄 Step 1: Creating synthetic dataset...")
        df = predictor.create_synthetic_dataset(num_samples=1000)
        
        # Step 2: Build model
        print("\n🏗️  Step 2: Building CNN model...")
        model = predictor.build_cnn_model()
        
        # Step 3: Prepare data splits
        print("\n📊 Step 3: Preparing data splits...")
        train_df, val_df, test_df = predictor.prepare_data(df)
        
        # Step 4: Train model
        print("\n🚀 Step 4: Training model...")
        history = predictor.train_model(train_df, val_df, epochs=30, batch_size=16)
        
        # Step 5: Evaluate model
        print("\n📈 Step 5: Evaluating model...")
        metrics, predictions = predictor.evaluate_model(test_df)
        
        # Step 6: Save model
        print("\n💾 Step 6: Saving model...")
        model_path = predictor.save_model("final_yield_prediction_model.h5")
        
        print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"📊 Final Results:")
        print(f"   RMSE: {metrics['rmse']:.3f} tons/ha")
        print(f"   MAE: {metrics['mae']:.3f} tons/ha")
        print(f"   R² Score: {metrics['r2_score']:.3f}")
        print(f"   MAPE: {metrics['mape']:.1f}%")
        print(f"\n📁 Output Files:")
        print(f"   📊 Dataset: data/crop_yield_dataset.csv")
        print(f"   🤖 Model: {model_path}")
        print(f"   📈 Results: results/evaluation_plots.png")
        print(f"   📋 Metrics: results/evaluation_metrics.json")
        
        return predictor, metrics
        
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        print(f"💡 Check the error log in results/training.log")
        raise

if __name__ == "__main__":
    predictor, results = run_complete_pipeline()