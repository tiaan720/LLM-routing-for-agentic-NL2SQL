import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.utils.chat_bot_arena_dataloader import ChatbotArenaDataLoader
from src.utils.logger import logger


class ModelRouter:
    def __init__(
        self,
        learning_rate=0.001,  # Reduced learning rate
        embedding_model="snowflake-arctic-embed:335m",
        embedding_provider="ollama",
        vector_store_path="data/chatbot_arena_inmemory_vectorstore_matrix_router.pkl",
        results_csv_path="data/model_arena_results_matrix_router.csv",
    ):
        # Use the same embedding model as the data loader
        self.results_csv_path = results_csv_path
        self.learning_rate = learning_rate
        self.eps = 1e-15  # Add small epsilon to prevent log(0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_dir = "router_models"
        self.model_name = "router_model"
        self.model_path = os.path.join(
            self.model_dir, f"{self.model_name}_{timestamp}.pkl"
        )

        # Create output directory for metrics and plots
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize training history for plotting
        self.training_history = {
            "epoch": [],
            "train_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
        }

        # Load vector store and embedding model from data loader
        loader = ChatbotArenaDataLoader(
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
        )
        loader.setup_vector_store(vector_store_path, recreate=False)
        self.vector_store = loader.vector_store
        self.embedding_model = loader.embedding_model

        # Build a query->embedding cache for fast lookup
        self.query_embedding_cache = {}
        if self.vector_store and hasattr(self.vector_store, "_docs"):
            for doc, emb in zip(self.vector_store._docs, self.vector_store._embeddings):
                self.query_embedding_cache[doc.page_content] = np.array(emb)

        # Autodetect embedding dimension
        if self.query_embedding_cache:
            # Use the first embedding in the cache
            self.model_embed_dim = len(next(iter(self.query_embedding_cache.values())))
        else:
            try:
                test_embedding = self.embedding_model.embed_query("test")
                self.model_embed_dim = len(test_embedding)
            except Exception as e:
                default_dim = 384
                logger.warning(
                    f"Could not auto-detect embedding dimension: {e}. Defaulting to {default_dim}."
                )
                self.model_embed_dim = default_dim

        # Better initialization - Xavier/He initialization
        self.W1 = np.random.randn(self.model_embed_dim, self.model_embed_dim) * np.sqrt(
            2.0 / self.model_embed_dim
        )
        self.w2 = np.random.randn(self.model_embed_dim) * np.sqrt(
            2.0 / self.model_embed_dim
        )
        self.model_embeddings = {}

    def embed_query(self, query: str) -> np.ndarray:
        """Get embedding for a query string using the precomputed vector store if available."""
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")

        # Use precomputed embedding if available
        if query in self.query_embedding_cache:
            return self.query_embedding_cache[query]
        # Fallback: compute embedding if not found
        embedding = self.embedding_model.embed_query(query)
        return np.array(embedding)

    def score_model(self, query_embedding: np.ndarray, model: str) -> float:
        projected_query = query_embedding @ self.W1
        model_embed = self.model_embeddings[model]
        score = self.w2 @ (model_embed * projected_query)
        return score

    def save_metrics_to_file(self, metrics, epoch=None, final=False):
        """Save detailed metrics to a text file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if final:
            filename = f"final_model_metrics_{timestamp}.txt"
        else:
            filename = f"epoch_{epoch}_metrics_{timestamp}.txt"

        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w") as f:
            f.write("=" * 60 + "\n")
            f.write(f"MATRIX ROUTER MODEL EVALUATION METRICS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if epoch is not None:
                f.write(f"Epoch: {epoch}\n")
            f.write(
                f"Model Type: {'Final Model' if final else 'Training Checkpoint'}\n"
            )
            f.write("\n")

            f.write("PERFORMANCE METRICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Accuracy:     {metrics.get('accuracy', 0):.4f}\n")
            f.write(f"Precision:    {metrics.get('precision', 0):.4f}\n")
            f.write(f"Recall:       {metrics.get('recall', 0):.4f}\n")
            f.write(f"F1 Score:     {metrics.get('f1', 0):.4f}\n")

            if hasattr(self, "training_history") and self.training_history["epoch"]:
                f.write("\n")
                f.write("TRAINING HISTORY:\n")
                f.write("-" * 30 + "\n")
                for i, epoch_num in enumerate(self.training_history["epoch"]):
                    f.write(f"Epoch {epoch_num:2d}: ")
                    f.write(
                        f"Train Loss: {self.training_history['train_loss'][i]:.4f}, "
                    )
                    f.write(
                        f"Train Acc: {self.training_history['train_accuracy'][i]:.4f}, "
                    )
                    f.write(
                        f"Val Acc: {self.training_history['val_accuracy'][i]:.4f}, "
                    )
                    f.write(f"Val F1: {self.training_history['val_f1'][i]:.4f}\n")

            f.write("\n")
            f.write("MODEL CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Learning Rate:     {self.learning_rate}\n")
            f.write(f"Embedding Dim:     {self.model_embed_dim}\n")
            f.write(
                f"Total Models:      {len(self.model_embeddings) if hasattr(self, 'model_embeddings') else 'N/A'}\n"
            )

        logger.info(f"Metrics saved to: {filepath}")
        return filepath

    def plot_training_curves(self, save_plots=True):
        """Create and save training curves using plotly."""
        if not hasattr(self, "training_history") or not self.training_history["epoch"]:
            logger.warning("No training history available for plotting")
            return

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Training Loss",
                "Accuracy Comparison",
                "Precision & Recall",
                "F1 Score",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        epochs = self.training_history["epoch"]

        # Plot 1: Training Loss
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=self.training_history["train_loss"],
                mode="lines+markers",
                name="Training Loss",
                line=dict(color="red", width=2),
            ),
            row=1,
            col=1,
        )

        # Plot 2: Accuracy Comparison
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=self.training_history["train_accuracy"],
                mode="lines+markers",
                name="Training Accuracy",
                line=dict(color="blue", width=2),
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=self.training_history["val_accuracy"],
                mode="lines+markers",
                name="Validation Accuracy",
                line=dict(color="green", width=2),
            ),
            row=1,
            col=2,
        )

        # Plot 3: Precision & Recall
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=self.training_history["val_precision"],
                mode="lines+markers",
                name="Validation Precision",
                line=dict(color="orange", width=2),
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=self.training_history["val_recall"],
                mode="lines+markers",
                name="Validation Recall",
                line=dict(color="purple", width=2),
            ),
            row=2,
            col=1,
        )

        # Plot 4: F1 Score
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=self.training_history["val_f1"],
                mode="lines+markers",
                name="Validation F1",
                line=dict(color="darkgreen", width=2),
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(
            title_text="Matrix Router Training Metrics",
            title_x=0.5,
            height=800,
            width=1200,
            showlegend=True,
            template="plotly_white",
        )

        # Update x-axis labels
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=2)

        # Update y-axis labels
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=2)
        fig.update_yaxes(title_text="Score", row=2, col=1)
        fig.update_yaxes(title_text="F1 Score", row=2, col=2)

        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(
                self.output_dir, f"training_curves_{timestamp}.pdf"
            )
            fig.write_image(plot_path, format="pdf", width=1200, height=800)
            logger.info(f"Training curves saved to: {plot_path}")

        fig.show()
        return fig

    def plot_final_metrics_summary(self, final_metrics, save_plots=True):
        """Create a summary plot of final model performance."""
        # Create a bar chart of final metrics
        metrics_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
        metrics_values = [
            final_metrics.get("accuracy", 0),
            final_metrics.get("precision", 0),
            final_metrics.get("recall", 0),
            final_metrics.get("f1", 0),
        ]

        # Define colors for each metric
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=metrics_names,
                    y=metrics_values,
                    marker_color=colors,
                    text=[f"{val:.3f}" for val in metrics_values],
                    textposition="auto",
                )
            ]
        )

        fig.update_layout(
            title="Final Model Performance Metrics",
            title_x=0.5,
            xaxis_title="Metrics",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            height=600,
            width=800,
            template="plotly_white",
            font=dict(size=14),
        )

        # Add a horizontal line at 0.5 for reference
        fig.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="gray",
            annotation_text="Random Baseline (0.5)",
        )

        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(
                self.output_dir, f"final_metrics_summary_{timestamp}.pdf"
            )
            fig.write_image(plot_path, format="pdf", width=800, height=600)
            logger.info(f"Final metrics summary saved to: {plot_path}")

        fig.show()
        return fig

    def save_model(self, metrics=None, epoch=None):
        """Save the trained model parameters."""
        os.makedirs(self.model_dir, exist_ok=True)

        # Create model name with metrics if available
        model_name = self.model_name
        if metrics and epoch is not None:
            accuracy = metrics.get("accuracy", 0)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{self.model_name}_acc{accuracy:.3f}_epoch{epoch}_{timestamp}"

        model_path = os.path.join(self.model_dir, f"{model_name}.pkl")

        model_state = {
            "W1": self.W1,
            "w2": self.w2,
            "model_embeddings": self.model_embeddings,
            "model_embed_dim": self.model_embed_dim,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "epoch": epoch,
            "training_history": getattr(self, "training_history", {}),
        }

        with open(model_path, "wb") as f:
            pickle.dump(model_state, f)
        logger.info(f"Model saved to: {model_path}")

    def load_model(self):
        """Load the most recent trained model parameters."""
        if not os.path.exists(self.model_dir):
            return False

        # Find most recent model file
        model_files = [
            f for f in os.listdir(self.model_dir) if f.startswith(self.model_name)
        ]
        if not model_files:
            return False

        latest_model = max(model_files)
        model_path = os.path.join(self.model_dir, latest_model)

        with open(model_path, "rb") as f:
            model_state = pickle.load(f)
        self.W1 = model_state["W1"]
        self.w2 = model_state["w2"]
        self.model_embeddings = model_state["model_embeddings"]
        self.model_embed_dim = model_state["model_embed_dim"]

        # Load training history if available
        if "training_history" in model_state:
            self.training_history = model_state["training_history"]

        logger.info(f"Loaded model from: {model_path}")
        logger.info(f"Model timestamp: {model_state.get('timestamp', 'unknown')}")
        return True

    def load_specific_model(self, model_filename: str) -> bool:
        """Load a specific model file from the models directory."""
        model_path = os.path.join(self.model_dir, model_filename)

        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False

        try:
            with open(model_path, "rb") as f:
                model_state = pickle.load(f)
            self.W1 = model_state["W1"]
            self.w2 = model_state["w2"]
            self.model_embeddings = model_state["model_embeddings"]
            self.model_embed_dim = model_state["model_embed_dim"]

            # Load training history if available
            if "training_history" in model_state:
                self.training_history = model_state["training_history"]

            logger.info(f"Loaded model from: {model_path}")
            logger.info(f"Model timestamp: {model_state.get('timestamp', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def evaluate(self, df: pd.DataFrame) -> dict:
        """Evaluate model performance on given dataset."""
        predictions = []
        actuals = []

        # Debug: Check if test queries are in cache
        queries_in_cache = 0
        queries_not_in_cache = 0

        for _, row in df.iterrows():
            try:
                # Check if query is in cache
                if row["query"] in self.query_embedding_cache:
                    queries_in_cache += 1
                else:
                    queries_not_in_cache += 1

                query_embedding = self.embed_query(row["query"])
                score_a = self.score_model(query_embedding, row["model_a"])
                score_b = self.score_model(query_embedding, row["model_b"])

                # Predict winner based on scores
                # Resolve the placeholder ('model_a'/'model_b') to the actual model name
                predicted_model_name = (
                    row["model_a"] if score_a > score_b else row["model_b"]
                )
                predictions.append(predicted_model_name)
                actuals.append(row["winner"])

            except KeyError as e:
                logger.error(
                    f"Model key not found during evaluation: {e}. This can happen if a model in the test set was not in the training set."
                )
                continue

        # Debug output
        logger.info(
            f"Evaluation: {queries_in_cache} queries in cache, {queries_not_in_cache} not in cache"
        )
        logger.info(f"Total predictions made: {len(predictions)}")
        logger.info(
            f"Predictions sample: {predictions[:5] if predictions else 'No predictions'}"
        )
        logger.info(f"Actuals sample: {actuals[:5] if actuals else 'No actuals'}")

        # Check prediction distribution
        if predictions:
            pred_counts = pd.Series(predictions).value_counts()
            logger.info(f"Prediction distribution: {dict(pred_counts)}")

        if not predictions:
            logger.error("No predictions made during evaluation!")
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        # Calculate metrics with multiclass average
        metrics = {
            "accuracy": accuracy_score(actuals, predictions),
            "precision": precision_score(
                actuals, predictions, average="weighted", zero_division=0
            ),
            "recall": recall_score(
                actuals, predictions, average="weighted", zero_division=0
            ),
            "f1": f1_score(actuals, predictions, average="weighted", zero_division=0),
        }

        return metrics

    def train(
        self,
        results_csv_path: str,
        test_size: float = 0.2,
        epochs: int = 4,
        use_class_weights=True,
    ):
        # Load and split data
        df = pd.read_csv(results_csv_path)

        # Debug: Verify data quality and check for duplicates
        print(f"\n=== Data Analysis ===")
        print(f"Total samples: {len(df)}")
        print(f"Unique queries: {df['query'].nunique()}")
        print(f"Duplicate queries: {len(df) - df['query'].nunique()}")
        print(f"Unique model_a values: {df['model_a'].nunique()}")
        print(f"Unique model_b values: {df['model_b'].nunique()}")
        print(f"Winner distribution:\n{df['winner'].value_counts()}")

        # Correct data splitting to prevent leakage
        unique_queries = df["query"].unique()
        train_queries, test_queries = train_test_split(
            unique_queries, test_size=test_size, random_state=42
        )
        train_df = df[df["query"].isin(train_queries)]
        test_df = df[df["query"].isin(test_queries)]

        print(
            f"\nTraining on {len(train_df)} samples, testing on {len(test_df)} samples"
        )

        # Check for overlap between train and test
        train_queries = set(train_df["query"].values)
        test_queries = set(test_df["query"].values)
        overlap = train_queries.intersection(test_queries)
        print(f"Query overlap between train and test: {len(overlap)} queries")
        if len(overlap) > 0:
            print(f"WARNING: Found query overlap! First few overlapping queries:")
            for i, query in enumerate(list(overlap)[:3]):
                print(f"  {i+1}. {query[:100]}...")

        # Initialize model embeddings
        unique_models = set(df["model_a"].unique()) | set(df["model_b"].unique())
        print(f"Unique models to train: {len(unique_models)}")
        print(f"Models: {list(unique_models)}")

        for model in tqdm(unique_models, desc="Initializing embeddings"):
            self.model_embeddings[model] = (
                np.random.normal(size=self.model_embed_dim) * 0.01
            )  # Smaller initialization

        # Calculate class weights for imbalanced data
        class_weights = {}
        if use_class_weights:
            winner_counts = train_df["winner"].value_counts()
            total_samples = len(train_df)
            # Calculate weight as inverse frequency, normalized
            class_weights = {
                model: total_samples / (len(unique_models) * count)
                for model, count in winner_counts.items()
            }
            print(f"\nUsing class weights to handle imbalance:")
            # print({k: round(v, 2) for k, v in class_weights.items()})

        # Initialize best model tracking
        best_accuracy = 0.0
        best_state = None

        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            with tqdm(
                train_df.iterrows(),
                total=len(train_df),
                desc=f"Epoch {epoch+1}/{epochs}",
            ) as pbar:
                for _, row in pbar:
                    try:
                        query_embedding = self.embed_query(row["query"])

                        # Get scores
                        score_a = self.score_model(query_embedding, row["model_a"])
                        score_b = self.score_model(query_embedding, row["model_b"])

                        # Convert winner to target
                        target = 1.0 if row["winner"] == row["model_a"] else 0.0

                        # Binary cross entropy loss with sigmoid
                        score_diff = score_a - score_b
                        predicted = 1 / (1 + np.exp(-score_diff))
                        predicted = np.clip(predicted, self.eps, 1 - self.eps)

                        loss = -(
                            target * np.log(predicted)
                            + (1 - target) * np.log(1 - predicted)
                        )

                        # Apply class weight
                        if use_class_weights:
                            winner_model = row["winner"]
                            weight = class_weights.get(winner_model, 1.0)
                            loss *= weight

                        # Compute gradients
                        grad_output = (
                            predicted - target
                        )  # Gradient of loss w.r.t score_diff

                        # Apply weight to gradient
                        if use_class_weights:
                            grad_output *= weight

                        # Gradient for model embeddings
                        projected_query = query_embedding @ self.W1

                        # Update model A embedding
                        grad_model_a = grad_output * (self.w2 * projected_query)
                        self.model_embeddings[row["model_a"]] -= (
                            self.learning_rate * grad_model_a
                        )

                        # Update model B embedding (negative because it's subtracted in score_diff)
                        grad_model_b = -grad_output * (self.w2 * projected_query)
                        self.model_embeddings[row["model_b"]] -= (
                            self.learning_rate * grad_model_b
                        )

                        # Update W1 matrix
                        model_a_embed = self.model_embeddings[row["model_a"]]
                        model_b_embed = self.model_embeddings[row["model_b"]]
                        grad_W1 = grad_output * np.outer(
                            query_embedding, self.w2 * (model_a_embed - model_b_embed)
                        )
                        self.W1 -= self.learning_rate * grad_W1

                        # Update w2 vector
                        grad_w2 = grad_output * (
                            (model_a_embed * projected_query)
                            - (model_b_embed * projected_query)
                        )
                        self.w2 -= self.learning_rate * grad_w2

                        total_loss += loss

                        # Track accuracy
                        if (predicted > 0.5 and target == 1) or (
                            predicted <= 0.5 and target == 0
                        ):
                            correct += 1
                        total += 1

                        # Update progress bar with current loss and accuracy
                        pbar.set_postfix(
                            {
                                "loss": f"{total_loss/(pbar.n+1):.4f}",
                                "acc": f"{correct/total:.3f}",
                            }
                        )

                    except ValueError as e:
                        logger.warning(f"Skipping row - {str(e)}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing row: {e}")
                        continue

            # Evaluate on test set with progress bar
            metrics = {}
            with tqdm(desc="Evaluating", total=1) as pbar:
                metrics = self.evaluate(test_df)
                pbar.update(1)

            # Store training history for plotting
            train_accuracy = correct / total
            avg_train_loss = total_loss / len(train_df)

            self.training_history["epoch"].append(epoch + 1)
            self.training_history["train_loss"].append(avg_train_loss)
            self.training_history["train_accuracy"].append(train_accuracy)
            self.training_history["val_accuracy"].append(metrics.get("accuracy", 0))
            self.training_history["val_precision"].append(metrics.get("precision", 0))
            self.training_history["val_recall"].append(metrics.get("recall", 0))
            self.training_history["val_f1"].append(metrics.get("f1", 0))

            logger.info(f"Epoch {epoch+1}")
            logger.info(f"Training Loss: {avg_train_loss:.4f}")
            logger.info(f"Training Accuracy: {train_accuracy:.3f}")
            logger.info(f"Validation Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Validation Precision: {metrics['precision']:.4f}")
            logger.info(f"Validation Recall: {metrics['recall']:.4f}")
            logger.info(f"Validation F1: {metrics['f1']:.4f}")

            # Save metrics for this epoch
            self.save_metrics_to_file(metrics, epoch=epoch + 1)

            # Save best model
            if metrics["accuracy"] > best_accuracy:
                best_accuracy = metrics["accuracy"]
                best_state = {
                    "W1": self.W1.copy(),
                    "w2": self.w2.copy(),
                    "model_embeddings": self.model_embeddings.copy(),
                    "metrics": metrics,
                }
                # Save the improved model
                self.save_model(metrics=metrics, epoch=epoch + 1)

        # Restore best model
        if best_state:
            self.W1 = best_state["W1"]
            self.w2 = best_state["w2"]
            self.model_embeddings = best_state["model_embeddings"]
            final_metrics = best_state["metrics"]
            logger.info(f"\nBest model metrics: {final_metrics}")
        else:
            # Use last epoch metrics if no improvement
            final_metrics = metrics

        # Save final model metrics
        self.save_metrics_to_file(final_metrics, final=True)

        # Generate and save training plots
        logger.info("Generating training curves...")
        self.plot_training_curves(save_plots=True)

        logger.info("Generating final metrics summary...")
        self.plot_final_metrics_summary(final_metrics, save_plots=True)

        # Save the final model
        self.save_model()

        # Print final summary
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETED - FINAL SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Final Accuracy:  {final_metrics['accuracy']:.4f}")
        logger.info(f"Final Precision: {final_metrics['precision']:.4f}")
        logger.info(f"Final Recall:    {final_metrics['recall']:.4f}")
        logger.info(f"Final F1 Score:  {final_metrics['f1']:.4f}")
        logger.info(f"Total Epochs:    {len(self.training_history['epoch'])}")
        logger.info(
            f"Best Epoch:      {self.training_history['epoch'][np.argmax(self.training_history['val_accuracy'])]}"
        )
        logger.info(f"Metrics and plots saved to: {self.output_dir}")
        logger.info("=" * 60)

    def route_query(self, query: str, threshold: float = 0.5) -> str:
        """Route a query to the best model based on learned embeddings."""
        if not self.model_embeddings:
            raise ValueError(
                "Model not trained yet. Call train() first or load_model()."
            )

        query_embedding = self.embed_query(query)
        scores = {}

        for model in self.model_embeddings:
            try:
                scores[model] = self.score_model(query_embedding, model)
            except Exception as e:
                logger.warning(f"Error scoring model {model}: {e}")
                continue

        if not scores:
            raise ValueError("No models could be scored")

        # Return the model with highest score
        best_model = max(scores.items(), key=lambda x: x[1])[0]
        logger.debug(f"Query: '{query[:50]}...' routed to: {best_model}")
        return best_model


if __name__ == "__main__":
    router = ModelRouter()
    router.train(router.results_csv_path, test_size=0.2, epochs=25)
    logger.info("Training completed and model saved.")
