import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from graphviz import Digraph
import os

class Visualizer:
    @staticmethod
    def plot_training_history(losses, save_path='static/images/training_history.png'):
        """Plot training loss over epochs"""
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_predictions(y_true, y_pred, save_path='static/images/predictions.png'):
        """Plot true vs predicted values"""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('True vs Predicted Values')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_neural_network_architecture(input_size=3, hidden_size=8, output_size=1):
        """Create neural network architecture visualization"""
        dot = Digraph(comment='Neural Network Architecture')
        dot.attr(rankdir='LR')  # Left to right layout

        # Input layer
        with dot.subgraph(name='cluster_input') as c:
            c.attr(label='Input Layer')
            for i in range(input_size):
                c.node(f'i{i}', f'Input {i+1}')

        # Hidden layer
        with dot.subgraph(name='cluster_hidden') as c:
            c.attr(label='Hidden Layer')
            for i in range(hidden_size):
                c.node(f'h{i}', f'Hidden {i+1}')

        # Output layer
        with dot.subgraph(name='cluster_output') as c:
            c.attr(label='Output Layer')
            c.node('o0', 'Sales')

        # Add connections
        for i in range(input_size):
            for h in range(hidden_size):
                dot.edge(f'i{i}', f'h{h}')
        for h in range(hidden_size):
            dot.edge(f'h{h}', 'o0')

        # Save the diagram
        dot.render('static/images/nn_architecture', format='png', cleanup=True)

    @staticmethod
    def plot_feature_importance(model, feature_names, save_path='static/images/feature_importance.png'):
        """Plot feature importance based on weights"""
        input_weights = np.abs(model.W1)
        importance = np.mean(input_weights, axis=1)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_names, y=importance)
        plt.title('Feature Importance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()