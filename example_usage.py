# examples/example_usage.py
from src.attention_visualizer import AttentionVisualizer

def main():
    # Initialize visualizer
    visualizer = AttentionVisualizer()
    
    # Example pairs
    examples = [
        {
            "query": "how do neural networks learn patterns",
            "document": "neural networks learn by adjusting connection weights through backpropagation, gradually minimizing prediction errors during training",
            "type": "Positive Example - Direct answer about neural network learning"
        },
        {
            "query": "how do neural networks learn patterns",
            "document": "Today, the weather pattern predicted by AI is sunny with a high of 75 degrees fahrenheit and light winds from the northwest",
            "type": "Negative Example - Completely irrelevant weather information"
        }
    ]
    
    # Analyze examples
    for i, example in enumerate(examples, 1):
        print(f"\n{'='*20} {example['type']} {'='*20}")
        visualizer.visualize(
            query=example["query"],
            document=example["document"]
        )

if __name__ == "__main__":
    main()
