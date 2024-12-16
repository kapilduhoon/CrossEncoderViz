from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from bertviz import head_view
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

class AttentionVisualizer:
    """
    A class to visualize and analyze neural network attention patterns in transformer models.
    
    This tool helps understand how transformer models process and relate different parts of 
    input text through their attention mechanisms.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        """
        Initialize the attention visualizer with a specific model.
        
        Args:
            model_name (str): The name or path of the pre-trained model to use
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.score_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Set models to evaluation mode
        self.model.eval()
        self.score_model.eval()
        
    def _prepare_input(self, query: str, document: str) -> Tuple[dict, List[str], int]:
        """
        Prepare the input for model processing.
        
        Args:
            query (str): The input query text
            document (str): The document text to analyze
            
        Returns:
            Tuple containing:
                - Model inputs
                - List of tokens
                - Position of the SEP token
        """
        inputs = self.tokenizer(
            query,
            document,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        sep_pos = tokens.index('[SEP]')
        
        return inputs, tokens, sep_pos
    
    def _get_model_outputs(self, inputs: dict) -> Tuple[torch.Tensor, float, List[torch.Tensor]]:
        """
        Get model outputs including attention patterns and relevance scores.
        
        Args:
            inputs (dict): Prepared model inputs
            
        Returns:
            Tuple containing:
                - Model outputs
                - Relevance score
                - List of attention tensors
        """
        with torch.no_grad():
            outputs = self.model(**inputs)
            score_outputs = self.score_model(**inputs)
            score = score_outputs.logits.item()
            all_attention = list(outputs.attentions)
            
        return outputs, score, all_attention
    
    def _analyze_layer_connections(
        self,
        attention: torch.Tensor,
        query_tokens: List[str],
        doc_tokens: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, str, float]]:
        """
        Analyze the strongest attention connections in a layer.
        
        Args:
            attention: Attention tensor for the layer
            query_tokens: List of query tokens
            doc_tokens: List of document tokens
            top_k: Number of top connections to return
            
        Returns:
            List of tuples containing (query_token, doc_token, attention_score)
        """
        avg_attention = attention.squeeze(0).mean(dim=0)
        q2d_attention = avg_attention[1:len(query_tokens)+1, len(query_tokens)+2:-1]
        
        flat_indices = torch.topk(q2d_attention.flatten(), top_k)
        
        connections = []
        for value, flat_idx in zip(flat_indices.values.tolist(), flat_indices.indices.tolist()):
            q_idx = flat_idx // len(doc_tokens)
            d_idx = flat_idx % len(doc_tokens)
            connections.append((query_tokens[q_idx], doc_tokens[d_idx], value))
            
        return connections
    
    def visualize(
        self,
        query: str,
        document: str,
        display_tokens: bool = True,
        key_layers: Optional[List[int]] = None
    ) -> None:
        """
        Visualize and analyze attention patterns between query and document.
        
        Args:
            query (str): The input query text
            document (str): The document text to analyze
            display_tokens (bool): Whether to display tokenized inputs
            key_layers (List[int], optional): Specific layers to analyze (0-based indexing)
        """
        # Prepare inputs and get outputs
        inputs, tokens, sep_pos = self._prepare_input(query, document)
        outputs, score, all_attention = self._get_model_outputs(inputs)
        
        # Print analysis header
        print("\n=== Query-Document Analysis ===")
        print(f"Query: {query}")
        print(f"Document: {document}")
        print(f"Cross-Encoder Relevance Score: {score:.4f}")
        
        if display_tokens:
            print("\nQuery tokens:", ' '.join(tokens[1:sep_pos]))
            print("Document tokens:", ' '.join(tokens[sep_pos+1:-1]))
        
        # Set default key layers if none provided
        if key_layers is None:
            key_layers = [0, 6, 11]  # Analyze layers 1, 7, and 12
        
        # Visualize attention for key layers
        for layer_idx in key_layers:
            print(f"\n=== Layer {layer_idx + 1} Analysis ===")
            
            plt.figure(figsize=(12, 8))
            attention = [all_attention[layer_idx]]
            head_view(
                attention=attention,
                tokens=tokens,
                sentence_b_start=sep_pos + 1
            )
            plt.title(f'Layer {layer_idx + 1} Neural Network Concept Understanding')
            plt.show()
            
            # Analyze conceptual relationships
            query_tokens = tokens[1:sep_pos]
            doc_tokens = tokens[sep_pos+1:-1]
            connections = self._analyze_layer_connections(
                attention[0],
                query_tokens,
                doc_tokens
            )
            
            print(f"\nKey Concept Connections in Layer {layer_idx + 1}:")
            for idx, (q_token, d_token, value) in enumerate(connections, 1):
                print(f" {idx}. '{q_token}' → '{d_token}': {value:.4f}")

