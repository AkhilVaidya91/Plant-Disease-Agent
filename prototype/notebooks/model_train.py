import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16, MobileNetV2, EfficientNetB0, ResNet50, InceptionV3, DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
import pandas as pd
import time
import os
import pickle

# Ensure TensorFlow uses GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Function to create a model using transfer learning
def create_model(base_model_name, input_shape, num_classes, dropout_rate=0.3):
    """
    Create a transfer learning model based on the specified architecture
    
    Args:
        base_model_name: Name of the base model to use
        input_shape: Shape of input images
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Model: Compiled keras model
    """
    if base_model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'InceptionV3':
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'DenseNet121':
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unsupported model: {base_model_name}")
    
    # We'll freeze the base model layers
    base_model.trainable = False

    ## unfreeze the last few layers of the base model
    for layer in base_model.layers[-5:]:
        layer.trainable = True
    
    # Add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    
    # Add a logistic layer for predictions
    if num_classes == 2:
        predictions = Dense(1, activation='sigmoid')(x)
    else:
        predictions = Dense(num_classes, activation='softmax')(x)
    
    # Final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

# Function to train a model and evaluate its performance
def train_and_evaluate_model(model_name, X_train, y_train, X_test, y_test, num_classes, 
                            epochs=15, batch_size=32, learning_rate=0.001,
                            save_dir='saved_models'):
    """
    Train and evaluate a model
    
    Args:
        model_name: Name of the model architecture
        X_train, y_train: Training data
        X_test, y_test: Test data
        num_classes: Number of output classes
        epochs: Maximum number of epochs to train
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        save_dir: Directory to save the trained model
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    print(f"\n{'-'*50}")
    print(f"Training {model_name}")
    print(f"{'-'*50}")
    
    # Create model
    model = create_model(model_name, X_train.shape[1:], num_classes)
    
    # Compile model
    if num_classes == 2:
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
    else:
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
    
    # Define callbacks
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(save_dir, f"{model_name}.h5"),
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    # Data augmentation for training
    datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
        zoom_range=0.05
    )
    datagen.fit(X_train)
    
    # Measure training time
    start_time = time.time()
    
    # Train the model
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[model_checkpoint, early_stopping, reduce_lr]
    )
    
    training_time = time.time() - start_time
    
    # Load the best model
    model.load_weights(os.path.join(save_dir, f"{model_name}.h5"))
    
    # Measure inference time
    start_time = time.time()
    predictions = model.predict(X_test)
    inference_time = (time.time() - start_time) / len(X_test)
    
    # Convert predictions to class labels
    if num_classes == 2:
        y_pred = (predictions > 0.5).astype("int32").flatten()
    else:
        y_pred = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    metrics = {}
    metrics['model_name'] = model_name
    metrics['accuracy'] = (y_pred == y_test).mean()
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics['precision'] = report['weighted avg']['precision']
    metrics['recall'] = report['weighted avg']['recall']
    metrics['f1_score'] = report['weighted avg']['f1-score']
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm
    
    # ROC curve and AUC for binary classification
    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, predictions)
        metrics['roc_auc'] = auc(fpr, tpr)
        metrics['fpr'] = fpr
        metrics['tpr'] = tpr
    else:
        # For multiclass, calculate ROC AUC using one-vs-rest approach
        metrics['roc_auc'] = []
        for i in range(num_classes):
            y_true_binary = (y_test == i).astype(int)
            if predictions.ndim > 1 and predictions.shape[1] > 1:
                y_score = predictions[:, i]
            else:
                y_score = predictions
            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            metrics['roc_auc'].append(auc(fpr, tpr))
        metrics['roc_auc'] = np.mean(metrics['roc_auc'])
    
    # Training history
    metrics['history'] = history.history
    
    # Model size
    model_size_mb = os.path.getsize(os.path.join(save_dir, f"{model_name}.h5")) / (1024 * 1024)
    metrics['model_size_mb'] = model_size_mb
    
    # Training and inference time
    metrics['training_time'] = training_time
    metrics['inference_time'] = inference_time * 1000  # Convert to milliseconds
    
    # Save the model in a separate format for easy loading
    model.save(os.path.join(save_dir, f"{model_name}_final.h5"))
    
    # Print key metrics
    print(f"\n{model_name} Evaluation:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Model Size: {metrics['model_size_mb']:.2f} MB")
    print(f"Training Time: {metrics['training_time']:.2f} seconds")
    print(f"Inference Time: {metrics['inference_time']:.2f} ms per sample")
    
    return metrics, model

# Function to plot training history
def plot_training_history(history, model_name):
    """
    Plot training and validation accuracy/loss
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    axes[0].plot(history['accuracy'], label='Training Accuracy')
    axes[0].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title(f'{model_name} - Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    
    # Loss plot
    axes[1].plot(history['loss'], label='Training Loss')
    axes[1].plot(history['val_loss'], label='Validation Loss')
    axes[1].set_title(f'{model_name} - Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    
    plt.tight_layout()
    return fig

# Function to plot ROC curve
def plot_roc_curve(fpr, tpr, roc_auc, model_name):
    """
    Plot ROC curve for binary classification
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curve')
    plt.legend(loc="lower right")
    return plt.gcf()

# Function to plot confusion matrix
def plot_confusion_matrix(cm, model_name, num_classes):
    """
    Plot confusion matrix
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(num_classes), 
                yticklabels=range(num_classes))
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return plt.gcf()

# Main function to train and evaluate multiple models
def evaluate_multiple_models(X_train, y_train, X_test, y_test, num_classes, 
                           save_dir='saved_models', visualizations_dir='visualizations'):
    """
    Train and evaluate multiple CNN models
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        num_classes: Number of output classes
        save_dir: Directory to save models
        visualizations_dir: Directory to save visualizations
    """
    # Create directories if they don't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(visualizations_dir):
        os.makedirs(visualizations_dir)
    
    # Model configurations to try
    model_configs = [
        {'name': 'VGG16', 'epochs': 15, 'batch_size': 32, 'learning_rate': 0.0001},
        {'name': 'MobileNetV2', 'epochs': 15, 'batch_size': 32, 'learning_rate': 0.0001},
        {'name': 'EfficientNetB0', 'epochs': 15, 'batch_size': 32, 'learning_rate': 0.0001},
        {'name': 'ResNet50', 'epochs': 15, 'batch_size': 32, 'learning_rate': 0.0001},
        {'name': 'InceptionV3', 'epochs': 15, 'batch_size': 32, 'learning_rate': 0.0001},
        {'name': 'DenseNet121', 'epochs': 15, 'batch_size': 32, 'learning_rate': 0.0001}
    ]
    
    all_metrics = []
    best_model = None
    best_accuracy = 0
    
    # Train and evaluate each model
    for config in model_configs:
        metrics, model = train_and_evaluate_model(
            model_name=config['name'],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            num_classes=num_classes,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            save_dir=save_dir
        )
        
        all_metrics.append(metrics)
        
        # Save visualizations
        # Training history
        fig_history = plot_training_history(metrics['history'], config['name'])
        fig_history.savefig(os.path.join(visualizations_dir, f"{config['name']}_history.png"))
        
        # Confusion matrix
        fig_cm = plot_confusion_matrix(metrics['confusion_matrix'], config['name'], num_classes)
        fig_cm.savefig(os.path.join(visualizations_dir, f"{config['name']}_confusion_matrix.png"))
        
        # ROC curve for binary classification
        if num_classes == 2:
            fig_roc = plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['roc_auc'], config['name'])
            fig_roc.savefig(os.path.join(visualizations_dir, f"{config['name']}_roc_curve.png"))
        
        plt.close('all')
        
        # Keep track of the best model
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            best_model = {'name': config['name'], 'model': model}
    
    # Save the best model
    if best_model:
        best_model_name = best_model['name']
        best_model['model'].save(os.path.join(save_dir, f"{best_model_name}_best.h5"))
        print(f"\nBest performing model: {best_model_name} with accuracy {best_accuracy:.4f}")
    
    # Create comparison table
    comparison_df = pd.DataFrame([{
        'Model': m['model_name'],
        'Accuracy': m['accuracy'],
        'Precision': m['precision'],
        'Recall': m['recall'],
        'F1 Score': m['f1_score'],
        'ROC AUC': m['roc_auc'],
        'Model Size (MB)': m['model_size_mb'],
        'Training Time (s)': m['training_time'],
        'Inference Time (ms)': m['inference_time']
    } for m in all_metrics])
    
    # Save comparison table
    comparison_df.to_csv(os.path.join(save_dir, 'model_comparison.csv'), index=False)
    
    # Save all metrics for later analysis
    with open(os.path.join(save_dir, 'all_metrics.pkl'), 'wb') as f:
        pickle.dump(all_metrics, f)
    
    return comparison_df, all_metrics

# Plot comparison bar charts
def plot_comparison_charts(comparison_df, save_dir='visualizations'):
    """
    Create bar charts comparing the performance of different models
    """
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    
    # Performance metrics comparison
    plt.figure(figsize=(15, 8))
    comparison_df[['Model'] + metrics_to_plot].set_index('Model').plot(kind='bar', width=0.8)
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_comparison.png'))
    
    # Resource usage comparison
    plt.figure(figsize=(15, 8))
    
    # Normalize model size and inference time for better visualization
    size_norm = comparison_df['Model Size (MB)'] / comparison_df['Model Size (MB)'].max()
    time_norm = comparison_df['Inference Time (ms)'] / comparison_df['Inference Time (ms)'].max()
    
    df_resource = pd.DataFrame({
        'Model': comparison_df['Model'],
        'Normalized Model Size': size_norm,
        'Normalized Inference Time': time_norm
    })
    
    df_resource.set_index('Model').plot(kind='bar', width=0.8)
    plt.title('Model Resource Usage Comparison (Normalized)')
    plt.ylabel('Normalized Value')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'resource_comparison.png'))
    
    plt.close('all')

# Example usage
if __name__ == "__main__":
    # Assuming X_train, X_test, y_train, y_test and k are already defined
    # X_train and X_test should be numpy arrays with shape (m, 224, 224, 3) and values between 0-1
    # y_train and y_test should be integer labels
    # k is the number of classes
    
    # Run the evaluation
    comparison_df, all_metrics = evaluate_multiple_models(
        X_train=X_train, 
        y_train=y_train, 
        X_test=X_test, 
        y_test=y_test, 
        num_classes=k
    )
    
    # Plot comparison charts
    plot_comparison_charts(comparison_df)
    
    # Print the comparison table
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))