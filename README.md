# Cross Encoder Attention Visualizer

This project provides tools to visualize and analyze attention patterns in Cross Encoder, helping understand how these models process and relate different parts of input text.

## Features

- Visualize attention patterns across different layers of transformer models
- Analyze query-document relationships through attention mechanisms
- Identify key conceptual connections between input texts
- Compare attention patterns between positive and negative examples
- Customizable visualization options


## Usage

Basic usage example:

```python
from src.attention_visualizer import AttentionVisualizer

# Initialize visualizer
visualizer = AttentionVisualizer()

# Analyze a query-document pair
visualizer.visualize(
    query="how do neural networks learn patterns",
    document="neural networks learn by adjusting connection weights through backpropagation"
)
```

See `examples/example_usage.py` for more detailed examples.

## Configuration

The default model used is "cross-encoder/ms-marco-MiniLM-L-12-v2", but you can specify a different model during initialization:

```python
visualizer = AttentionVisualizer(model_name="your-preferred-model")
```

## Output

The visualizer provides:
1. Attention visualization plots for key layers
2. Cross-encoder relevance scores
3. Token-level analysis
4. Top attention connections between query and document terms

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- BertViz
- Matplotlib

## License

This project is licensed under the MIT License - see the LICENSE file for details.
