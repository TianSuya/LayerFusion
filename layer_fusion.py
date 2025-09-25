import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import gc
from copy import deepcopy
import random
from dataclasses import dataclass
import warnings
from probe_generator import ProbeGenerator, ProbeData
from transformers.masking_utils import create_causal_mask
from LayerFusion import LaCO, LayerAvg, Balance

warnings.filterwarnings("ignore", category=UserWarning)

def has_overlap(interval1: Tuple[int, int], interval2: Tuple[int, int]) -> bool:
    return interval1[0] <= interval2[1] and interval2[0] <= interval1[1]

def has_overlap_with_set(interval: Tuple[int, int], intervals: List[Tuple[int, int]]) -> bool:
    for interval2 in intervals:
        if has_overlap(interval, interval2):
            return True
    return False

@dataclass
class ContinuousLayerBlock:
    start_layer: int
    end_layer: int
    size: int
    input_similarity: float
    output_similarity: float
    combined_score: float

    def __post_init__(self):
        self.size = self.end_layer - self.start_layer + 1


class ContinuousLayerSelector:
    
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
    
    def generate_continuous_blocks(self, 
                                block_size: int = 5,
                                 compression_ratio: float = 0.25) -> List[Tuple[int, int]]:
        
        target_layers_to_compress = int(self.num_layers * compression_ratio)
        assert target_layers_to_compress % (block_size - 1) == 0
        
        all_combinations = []
        
        for start_layer in range(self.num_layers - block_size + 1):
            end_layer = start_layer + block_size - 1
            all_combinations.append((start_layer, end_layer))

        return all_combinations, target_layers_to_compress


class ContinuousLayerSimilarityCalculator:
    
    def __init__(self, 
                 model: AutoModelForCausalLM,
                 tokenizer: AutoTokenizer,
                 device: str = 'cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_layers = len(model.model.layers)
    
    def collect_layer_io_states(self, 
                              representative_inputs: torch.Tensor,
                              attention_mask: torch.Tensor = None) -> Dict[int, Dict[str, torch.Tensor]]:
        
        layer_io_states = {}
        
        current_hidden_states = representative_inputs.clone()
        batch_size = current_hidden_states.shape[0]
        seq_len = current_hidden_states.shape[1]
        
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=self.device)
        
        causal_mask = create_causal_mask(
            config=self.model.config,
            input_embeds=current_hidden_states,
            attention_mask=attention_mask,
            cache_position=torch.arange(current_hidden_states.shape[1], device=self.device),
            past_key_values=None,
            position_ids=position_ids,
        )
        
        with torch.no_grad():
            for layer_idx in tqdm(range(self.num_layers), desc="Collect Input and Output"):
                layer = self.model.model.layers[layer_idx]
                
                layer_input = current_hidden_states.clone()
                
                position_embeddings = self.model.model.rotary_emb(current_hidden_states, position_ids)
                
                layer_output = layer(
                    hidden_states=current_hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                    output_attentions=False
                )
                
                if isinstance(layer_output, tuple):
                    current_hidden_states = layer_output[0]
                else:
                    current_hidden_states = layer_output
                
                layer_io_states[layer_idx] = {
                    'input': layer_input[:, -1, :].detach().cpu(),  # [batch_size, hidden_dim]
                    'output': current_hidden_states[:, -1, :].detach().cpu()
                }
                
                torch.cuda.empty_cache()
        
        return layer_io_states
    
    def calculate_block_similarity(self, 
                                 layer_io_states: Dict[int, Dict[str, torch.Tensor]],
                                 start_layer: int, 
                                 end_layer: int,
                                 similarity_metric: str = 'cosine') -> Tuple[float, float]:
        
        block_input = layer_io_states[start_layer]['input']
        block_output = layer_io_states[end_layer]['output']
        
        if similarity_metric == 'cosine':
            similarities = []
            for i in range(block_input.shape[0]):
                sim = F.cosine_similarity(
                    block_input[i].unsqueeze(0), 
                    block_output[i].unsqueeze(0), 
                    dim=1
                ).item()
                similarities.append(sim)
            
            input_output_similarity = np.mean(similarities)

            return input_output_similarity, input_output_similarity
            
        elif similarity_metric == 'euclidean':
            distances = []
            for i in range(block_input.shape[0]):
                dist = torch.norm(block_input[i] - block_output[i], p=2).item()
                distances.append(dist)
            
            avg_distance = np.mean(distances)
            similarity = 1.0 / (1.0 + avg_distance)
            return similarity, similarity
            
        elif similarity_metric == 'cka':
            try:
                # CKA相似度
                similarity = cka_similarity(block_input, block_output, kernel_type='linear')
                return similarity, similarity
            except Exception as e:
                return self.calculate_block_similarity(
                    layer_io_states, start_layer, end_layer, 'cosine'
                )
        else:
            raise ValueError(f"Unsupported similarity metric: {similarity_metric}")


class ContinuousLayerFusion:
    
    def __init__(self, model: AutoModelForCausalLM, args=None):
        self.model = model
        self.original_layers = list(model.model.layers)
        self.layer_io_states = None  
        self.args = args 
    
    def set_layer_io_states(self, layer_io_states: Dict[int, Dict[str, torch.Tensor]]):
        self.layer_io_states = layer_io_states
    
    
    def compute_layer_functionality_weights(self,
                                          weight_list: List[torch.Tensor],
                                          start_layer: int,
                                          end_layer: int,
                                          similarity_metric: str = 'cosine') -> torch.Tensor:
        
        functionality_scores = []
        
        for layer_idx in range(start_layer, end_layer + 1):
            layer_input = self.layer_io_states[layer_idx]['input']   # [batch_size, hidden_dim]
            layer_output = self.layer_io_states[layer_idx]['output'] # [batch_size, hidden_dim]
            similarities = []
            for i in range(layer_input.shape[0]):
                sim = F.cosine_similarity(
                    layer_input[i].unsqueeze(0), 
                    layer_output[i].unsqueeze(0), 
                    dim=1
                ).item()
                similarities.append(sim)
            similarities = torch.tensor(similarities)
            layer_similarity = torch.mean(similarities)
            functionality_score = 1.0 - layer_similarity
            functionality_scores.append(functionality_score)
        
        functionality_scores = torch.stack(functionality_scores, dim=0)
        print('Functionality Scores', functionality_scores)

        enhanced_scores = functionality_scores / torch.sum(functionality_scores)
        print('Enhanced Scores', enhanced_scores)
        enhanced_scores = torch.pow(enhanced_scores, 2)
        enhanced_scores = enhanced_scores / torch.sum(enhanced_scores)

        print('Normalized Scores', enhanced_scores)

        enhanced_scores = F.softmax(enhanced_scores, dim=0)

        print('Softmax Scores', enhanced_scores)
        
        enhanced_scores = [s.item() for s in enhanced_scores]
        
        target_weight = torch.zeros_like(weight_list[0])
        for i, weight in enumerate(enhanced_scores):
            target_weight += weight * weight_list[i]
        
        return target_weight
    
    def _enhance_weight_differences(self, scores: List[float], power: float = 2.0) -> List[float]:

        if not scores or all(s == 0 for s in scores):
            return scores
        
        max_score = max(scores)
        min_score = min(scores)
        if max_score == min_score:
            return scores
        
        normalized_scores = [(s - min_score) / (max_score - min_score) for s in scores]

        print('normalized_scores', normalized_scores)

        enhanced_scores = [s ** power for s in normalized_scores]
        
        enhanced_scores = [max(s, 1e-8) for s in enhanced_scores]
        
        return enhanced_scores
    
    def fuse_continuous_blocks(self, 
                             args,
                             selected_blocks: List[Tuple[int, int]],
                             fusion_method: str = 'avg') -> AutoModelForCausalLM:

        print(f"Start fusing {len(selected_blocks)} continuous layer blocks, method: {fusion_method}")
        
        fused_model = type(self.model)(self.model.config)
        fused_model.load_state_dict(self.model.state_dict())
        
        layers_to_remove = set()
        fusion_mapping = {}  # {target_layer_idx: [source_layer_indices]}
        
        for block in selected_blocks:
            print(f"Fuse block: layer {block[0]}-{block[1]}, score: {block[2]:.4f}")
            
            target_layer_idx = block[0]
            source_layer_indices = list(range(block[0] + 1, block[1] + 1))
            
            fusion_mapping[target_layer_idx] = source_layer_indices
            layers_to_remove.update(source_layer_indices)
        
        new_layers = list(fused_model.model.layers)
        
        for target_idx, source_indices in fusion_mapping.items():
            target_layer = new_layers[target_idx]
            source_layers = [new_layers[idx] for idx in source_indices]

            self._fusion(
                target_layer=target_layer,
                source_layers=source_layers,
                method=args.fusion_method,
                ratio=args.coeff,
                target_method=args.target_method,
                start_layer=target_idx,
                end_layer=target_idx + args.block_size - 1,
                similarity_metric=getattr(args, 'similarity_metric', 'cosine')
            )
        
        final_layers = []
        for i, layer in enumerate(new_layers):
            if i not in layers_to_remove:
                final_layers.append(layer)
        
        fused_model.model.layers = nn.ModuleList(final_layers)
        fused_model.config.num_hidden_layers = len(final_layers)
        
        compression_achieved = 1 - len(final_layers) / len(self.original_layers)
        print(f"Layer fusion completed: {len(self.original_layers)} -> {len(final_layers)} layers")
        print(f"Actual compression ratio: {compression_achieved:.2%}")
        
        return fused_model

    def _fusion(self,
        target_layer: nn.Module,
        source_layers: List[nn.Module],
        method: str,
        ratio: float,
        target_method: str,
        start_layer: int = None,
        end_layer: int = None,
        similarity_metric: str = 'cosine',
    ):
        target_dict = target_layer.state_dict()

        for name in target_dict.keys():
            print(name)

            if name in ['input_layernorm.weight', 'post_attention_layernorm.weight']:
                continue

            weight_list = []
            weight_list.append(target_dict[name].detach().clone())
            for source_layer in source_layers:
                weight_list.append(source_layer.state_dict()[name].detach().clone())
            elif target_method == 'mean':
                target_weight = torch.stack(weight_list).mean(dim=0)
            elif target_method == 'functionality_weighted':
                target_weight = self.compute_layer_functionality_weights(
                    weight_list, start_layer, end_layer, similarity_metric
                )
            else:
                raise ValueError(f"Unsupported target layer method: {target_method}")

            if method == 'laco':
                fused_weight = LaCO(weight_list=weight_list, target_weight=target_weight, laco_ratio=ratio)

            elif method == 'balance':
                fused_weight = Balance(weight_list=weight_list, target_weight=target_weight, ratio=ratio, balance_ratio=self.args.pruning_percent)

            elif method == 'avg':
                fused_weight = LayerAvg(weight_list=weight_list)

            else:
                raise ValueError(f"Unsupported fusion method: {method}")

            target_dict[name] = fused_weight

        target_layer.load_state_dict(target_dict)
        

class ContinuousLayerFusionSystem:
    
    def __init__(self,
                 args,
                 model_path: str,
                 device: str = 'cuda',
                 probe_language: str = 'zh'):
        
        self.model_path = model_path
        self.device = device
        self.probe_language = probe_language
        
        print(f"Loading model: {model_path}")
        self.model, self.tokenizer = self._load_model()
        self.block_size = args.block_size
        
        self.probe_generator = ProbeGenerator(language=probe_language)
        self.layer_selector = ContinuousLayerSelector(len(self.model.model.layers))
        self.similarity_calculator = ContinuousLayerSimilarityCalculator(
            self.model, self.tokenizer, device
        )
        self.fusion_executor = ContinuousLayerFusion(self.model, args)
        
        self.probe_set = None
        self.representative_inputs = None
        self.attention_mask = None
        self.layer_io_states = None
        self.candidate_blocks = []
        self.selected_blocks = []
        
        print("Continuous layer fusion system initialized")
    
    def _load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True,
                padding_side='left',
                use_fast=False
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map='auto',
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            model.eval()
            
            print(f"Model loaded successfully: {model.config.num_hidden_layers} layers")
            return model, tokenizer
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            raise
    
    def generate_representative_inputs(self, 
                                     num_probes: int = 100,
                                     max_length: int = 128) -> torch.Tensor:
        print(f"Generating {num_probes} representative inputs...")
        
        self.probe_set = self.probe_generator.generate_diverse_probe_set(num_probes)
        
        probe_texts = [probe.text for probe in self.probe_set]
        
        batch_texts = probe_texts
        
        encoded = self.tokenizer(
            batch_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        with torch.no_grad():
            embeddings = self.model.model.embed_tokens(input_ids)
            # print(embeddings.shape)
        
        self.representative_inputs = embeddings
        self.attention_mask = attention_mask
        
        print(f"Representative inputs generated: {self.representative_inputs.shape}")
        return self.representative_inputs
    
    def run_continuous_fusion_analysis(self,
                                     args,
                                     compression_ratio: float = 0.25,
                                     similarity_metric: str = 'cosine',
                                     num_probes: int = 50,
                                     selection_strategy: str = 'highest_similarity') -> Dict:
        
        results = {}
        
        print(f"\nStep 1: Generating representative inputs...")
        self.generate_representative_inputs(num_probes)
        results['num_probes'] = num_probes
        results['input_shape'] = list(self.representative_inputs.shape)
        
        print(f"\nStep 2: Collecting layer input and output states...")
        self.layer_io_states = self.similarity_calculator.collect_layer_io_states(
            self.representative_inputs.to(self.device),
            self.attention_mask.to(self.device)
        )
        results['layers_processed'] = len(self.layer_io_states)
        
        print(f"\nStep 3: Generating continuous layer block candidates...")
        block_combinations, self.target_layers_to_compress = self.layer_selector.generate_continuous_blocks(
            compression_ratio=compression_ratio,
            block_size=self.block_size
        )
        results['candidate_combinations'] = len(block_combinations)
        
        print(f"\nStep 4: Evaluating similarity of each candidate combination...")
        best_combination = None
        best_score = -float('inf')

        combination_score = []
            
        for start_layer, end_layer in block_combinations:
            input_sim, output_sim = self.similarity_calculator.calculate_block_similarity(
                self.layer_io_states, start_layer, end_layer, similarity_metric
            )
            
            combined_score = (input_sim + output_sim) / 2

            combination_score.append((start_layer, end_layer, combined_score))
            
            if selection_strategy == 'highest_similarity':
                combination_score.sort(key=lambda x: x[2], reverse=True)
            elif selection_strategy == 'lowest_similarity':
                combination_score.sort(key=lambda x: x[2], reverse=False)

        selected_blocks = []
        self.blocks_num = int(self.target_layers_to_compress / (self.block_size - 1))
        for item in combination_score:
            if not has_overlap_with_set(item, selected_blocks):
                selected_blocks.append(item)
            if len(selected_blocks) >= self.blocks_num:
                break

        
        self.selected_blocks = selected_blocks

        print(selected_blocks)
        
        return results
    
    def execute_fusion(self,
                      args,
                      fusion_method: str = 'avg',
                      output_model_path: Optional[str] = None) -> AutoModelForCausalLM:
        
        
        print(f"Executing continuous layer fusion, method: {fusion_method}")
        
        if hasattr(self, 'layer_io_states') and self.layer_io_states is not None:
            self.fusion_executor.set_layer_io_states(self.layer_io_states)
        
        fused_model = self.fusion_executor.fuse_continuous_blocks(
            args,self.selected_blocks, fusion_method
        )

        if output_model_path:
            os.makedirs(output_model_path, exist_ok=True)
            fused_model.save_pretrained(output_model_path)
            self.tokenizer.save_pretrained(output_model_path)
        
        return fused_model


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Continuous layer fusion system')
    parser.add_argument('--model_path', type=str, default='./llama3_1',
                       help='Model path')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Computing device')
    parser.add_argument('--probe_language', type=str, default='en', help='Probe language')
    parser.add_argument('--compression_ratio', type=float, default=0.25,
                       help='Compression ratio')
    parser.add_argument('--similarity_metric', type=str, default='cosine',
                       choices=['cosine', 'euclidean', 'cka'],
                       help='Similarity metric')
    parser.add_argument('--num_probes', type=int, default=50,
                       help='Number of probes')
    parser.add_argument('--fusion_method', type=str, default='avg',
                       help='Fusion method')
    parser.add_argument('--output_model_path', type=str, default=None,
                       help='Output model path')
    parser.add_argument('--analysis_only', action='store_true',
                       help='Only perform analysis, not execution')
    parser.add_argument('--coeff', type=float, default=0.3,
                       help='Fusion ratio')
    parser.add_argument('--block_size', type=int, default=5,
                       help='Continuous layer block size')
    parser.add_argument('--target_method', type=str, default='mean')
    
    args = parser.parse_args()
    
    # Create fusion system
    fusion_system = ContinuousLayerFusionSystem(
        args=args,
        model_path=args.model_path,
        device=args.device,
        probe_language=args.probe_language
    )
    
    # Run analysis
    results = fusion_system.run_continuous_fusion_analysis(
        args=args,
        compression_ratio=args.compression_ratio,
        similarity_metric=args.similarity_metric,
        num_probes=args.num_probes
    )
    
    # Execute fusion (if needed)
    if not args.analysis_only:
        output_path = args.output_model_path
        if output_path is None:
            model_name = os.path.basename(args.model_path.rstrip('/'))
            output_path = f"./compressed_model/{args.fusion_method}-blocksize{args.block_size}-coeff{args.coeff}-prune{args.pruning_percent}-target{args.target_method}-compression{args.compression_ratio:.2f}"
        
        fused_model = fusion_system.execute_fusion(
            args=args,
            fusion_method=args.fusion_method,
            output_model_path=output_path
        )
        
        print(f"Continuous layer fusion completed! Model saved to: {output_path}")
    
    print("Continuous layer fusion analysis completed!")


if __name__ == '__main__':
    main()
