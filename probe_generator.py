import torch
import random
from typing import List, Dict, Tuple
import json
from dataclasses import dataclass


@dataclass
class ProbeData:
    text: str
    category: str
    subcategory: str
    expected_capability: str


class ProbeGenerator:
    
    def __init__(self, language: str = 'zh'):
        self.language = language
        self.probe_templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict:
        if self.language == 'en':
            return {
                'syntax': {
                    'simple_sentence': [
                        "The cat is sleeping on the couch.",
                        "Students are studying in the library.",
                        "Birds fly in the blue sky.",
                        "The teacher explains the lesson clearly.",
                        "Flowers bloom in spring."
                    ],
                    'complex_sentence': [
                        "Although it was raining, they continued their journey.",
                        "Because he studied hard, he passed the exam.",
                        "If you work diligently, you will succeed.",
                        "While she was cooking, the phone rang.",
                        "Since it's late, we should go home."
                    ],
                    'question_sentence': [
                        "How are you doing today?",
                        "When did the meeting start?",
                        "Why did you choose this option?",
                        "Where can I find this book?",
                        "Who can help solve this problem?"
                    ]
                },
                'semantics': {
                    'word_relations': [
                        "Apples and oranges are both fruits.",
                        "Doctors and nurses work in hospitals.",
                        "Cats and dogs are pets.",
                        "Spring, summer, autumn, and winter are seasons.",
                        "Red, yellow, and blue are primary colors."
                    ],
                    'metaphor': [
                        "Time is money.",
                        "Her voice is music to my ears.",
                        "He has a heart of stone.",
                        "Knowledge is power.",
                        "Life is a journey."
                    ]
                },
                'reasoning': {
                    'logical_inference': [
                        "All birds have wings. Penguins are birds. Therefore, penguins have wings.",
                        "If today is Monday, then tomorrow is Tuesday. Today is Monday.",
                        "Only students who study hard get good grades. John got good grades."
                    ],
                    'common_sense': [
                        "Fish live in water and cannot breathe on land.",
                        "Fire is hot and ice is cold.",
                        "The sun rises in the east and sets in the west.",
                        "Heavy objects fall down due to gravity.",
                        "Plants need sunlight for photosynthesis."
                    ]
                },
                'knowledge': {
                    'factual': [
                        "The capital of France is Paris.",
                        "The Earth orbits around the Sun.",
                        "Water has the chemical formula H2O.",
                        "A year has 365 days (or 366 in leap years).",
                        "The speed of light is approximately 300,000 km/s."
                    ]
                },
                'mathematics': {
                    'arithmetic': [
                        "3 + 5 = 8",
                        "12 × 4 = 48",
                        "100 ÷ 5 = 20",
                        "15 - 7 = 8"
                    ],
                    'word_problems': [
                        "John has 10 apples and eats 3. How many are left?",
                        "A class has 30 students, 15 are boys. How many are girls?",
                        "A car travels 60 km/h. How far in 3 hours?"
                    ]
                }
            }
    
    def generate_probe_set(self, num_samples_per_category: int = 10) -> List[ProbeData]:
        probe_set = []
        
        for category, subcategories in self.probe_templates.items():
            for subcategory, templates in subcategories.items():
                sampled_templates = random.sample(
                    templates, 
                    min(num_samples_per_category, len(templates))
                )
                
                for template in sampled_templates:
                    probe_data = ProbeData(
                        text=template,
                        category=category,
                        subcategory=subcategory,
                        expected_capability=f"{category}_{subcategory}"
                    )
                    probe_set.append(probe_data)
        
        return probe_set
    
    def generate_diverse_probe_set(self, total_samples: int = 200) -> List[ProbeData]:
        all_probes = []
        
        categories = list(self.probe_templates.keys())
        samples_per_category = total_samples // len(categories)
        
        for category in categories:
            subcategories = list(self.probe_templates[category].keys())
            samples_per_subcategory = max(1, samples_per_category // len(subcategories))
            
            for subcategory in subcategories:
                templates = self.probe_templates[category][subcategory]
                num_samples = min(samples_per_subcategory, len(templates))
                
                sampled_templates = random.sample(templates, num_samples)
                
                for template in sampled_templates:
                    probe_data = ProbeData(
                        text=template,
                        category=category,
                        subcategory=subcategory,
                        expected_capability=f"{category}_{subcategory}"
                    )
                    all_probes.append(probe_data)
        
        random.shuffle(all_probes)
        return all_probes[:total_samples]
    
    def save_probe_set(self, probe_set: List[ProbeData], save_path: str):
        probe_dict_list = []
        for probe in probe_set:
            probe_dict_list.append({
                'text': probe.text,
                'category': probe.category,
                'subcategory': probe.subcategory,
                'expected_capability': probe.expected_capability
            })
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(probe_dict_list, f, ensure_ascii=False, indent=2)
        
        print(f"Probe data set saved to: {save_path}")
        print(f"Total number of samples: {len(probe_set)}")
        
        category_counts = {}
        for probe in probe_set:
            if probe.category not in category_counts:
                category_counts[probe.category] = 0
            category_counts[probe.category] += 1
        
        print("Number of samples per category:")
        for category, count in category_counts.items():
            print(f"  {category}: {count}")
    
    def load_probe_set(self, load_path: str) -> List[ProbeData]:
        with open(load_path, 'r', encoding='utf-8') as f:
            probe_dict_list = json.load(f)
        
        probe_set = []
        for probe_dict in probe_dict_list:
            probe_data = ProbeData(
                text=probe_dict['text'],
                category=probe_dict['category'],
                subcategory=probe_dict['subcategory'],
                expected_capability=probe_dict['expected_capability']
            )
            probe_set.append(probe_data)
        
        print(f"Loaded {len(probe_set)} probe samples from {load_path}")
        return probe_set


def main():

    # Create English probe generator
    en_generator = ProbeGenerator(language='en')
    en_probe_set = en_generator.generate_diverse_probe_set(total_samples=100)
    en_generator.save_probe_set(en_probe_set, 'probes_en.json')
    
    print("\nProbe generation completed!")


if __name__ == '__main__':
    main()
