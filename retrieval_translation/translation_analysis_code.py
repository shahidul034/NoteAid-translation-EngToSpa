import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
import glob

# Set visual style
plt.style.use('ggplot')
sns.set_palette("colorblind")
sns.set_context("talk")

class TranslationExperimentAnalyzer:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.data = None
        self.metrics_of_interest = [
            'BLEU', 'CHRF++', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 
            'BERTScore_P', 'BERTScore_R', 'BERTScore_F', 'COMET'
        ]
        
    def load_all_metrics(self):
        """Load all metrics files from the directory structure"""
        all_data = []
        
        # Pattern: [type]_[method]_T[template]_[shots]_metrics.json
        pattern = re.compile(r'(raw|rerank)_([a-z0-9_]+)_T([0-9]+)_([0-9]+)_metrics\.json')
        
        # Walk through the directory structure
        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith("_metrics.json"):
                    match = pattern.match(file)
                    if match:
                        result_type = match.group(1)  # 'raw' or 'rerank'
                        method = match.group(2)
                        template = int(match.group(3))
                        shots = int(match.group(4))
                        
                        # Parse embedding source and model from path
                        path_parts = Path(root).parts
                        
                        # Look for "sentence" or "scielo" in the path
                        dataset = next((part for part in path_parts if part in ["sentence", "scielo"]), "unknown")
                        
                        # Look for embedding source and model
                        embedding_info = next((part for part in path_parts if part.startswith("local_") or part.startswith("openai_")), "unknown")
                        if embedding_info != "unknown":
                            embedding_source, embedding_model = embedding_info.split("_", 1)
                            # Clean up model name
                            embedding_model = embedding_model.replace("_", " ")
                        else:
                            embedding_source = "unknown"
                            embedding_model = "unknown"
                        
                        # Load the metrics
                        try:
                            file_path = os.path.join(root, file)
                            with open(file_path, 'r') as f:
                                metrics = json.load(f)
                                
                            # Add metadata to metrics
                            metrics_with_meta = {
                                'result_type': result_type,
                                'method': method,
                                'template': f"T{template}",
                                'shots': shots,
                                'dataset': dataset,
                                'embedding_source': embedding_source,
                                'embedding_model': embedding_model,
                                **metrics
                            }
                            all_data.append(metrics_with_meta)
                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")
        
        # Convert to DataFrame
        self.data = pd.DataFrame(all_data)
        return self.data
    
    def analyze_by_component(self):
        """Generate analysis by different components of the system"""
        if self.data is None:
            self.load_all_metrics()
            
        results = {
            'method_impact': self._analyze_method_impact(),
            'template_impact': self._analyze_template_impact(),
            'shot_impact': self._analyze_shot_impact(),
            'rerank_impact': self._analyze_rerank_impact(),
            'embedder_impact': self._analyze_embedder_impact(),
            'interactions': self._analyze_interactions()
        }
        
        return results
    
    def _analyze_method_impact(self):
        """Analyze impact of retrieval method on performance"""
        # Group by method and calculate mean metrics
        method_impact = self.data.groupby('method')[self.metrics_of_interest].mean().reset_index()
        return method_impact
    
    def _analyze_template_impact(self):
        """Analyze impact of prompt template on performance"""
        # Group by template and calculate mean metrics
        template_impact = self.data.groupby('template')[self.metrics_of_interest].mean().reset_index()
        return template_impact
    
    def _analyze_shot_impact(self):
        """Analyze impact of number of shots on performance"""
        # Group by shots and calculate mean metrics
        shot_impact = self.data.groupby('shots')[self.metrics_of_interest].mean().reset_index()
        return shot_impact
    
    def _analyze_rerank_impact(self):
        """Analyze impact of reranking on performance"""
        # Compare raw vs reranked results
        rerank_impact = self.data.groupby('result_type')[self.metrics_of_interest].mean().reset_index()
        return rerank_impact
    
    def _analyze_embedder_impact(self):
        """Analyze impact of embedding source and model on performance"""
        # Group by embedding source and model
        embedder_impact = self.data.groupby(['embedding_source', 'embedding_model'])[self.metrics_of_interest].mean().reset_index()
        return embedder_impact
    
    def _analyze_interactions(self):
        """Analyze interactions between different components"""
        # Method x Template
        method_template = self.data.groupby(['method', 'template'])[['BLEU', 'COMET']].mean().reset_index()
        
        # Method x Shots
        method_shots = self.data.groupby(['method', 'shots'])[['BLEU', 'COMET']].mean().reset_index()
        
        # Method x Result Type (raw vs reranked)
        method_result_type = self.data.groupby(['method', 'result_type'])[['BLEU', 'COMET']].mean().reset_index()
        
        # Template x Shots
        template_shots = self.data.groupby(['template', 'shots'])[['BLEU', 'COMET']].mean().reset_index()
        
        # Template x Result Type
        template_result_type = self.data.groupby(['template', 'result_type'])[['BLEU', 'COMET']].mean().reset_index()
        
        return {
            'method_template': method_template,
            'method_shots': method_shots,
            'method_result_type': method_result_type,
            'template_shots': template_shots,
            'template_result_type': template_result_type
        }
    
    def generate_plots(self, output_dir=None):
        """Generate visualization plots for the analysis"""
        if self.data is None:
            self.load_all_metrics()
            
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        plots = {}
        
        # 1. Method comparison
        plots['method_comparison'] = self._plot_method_comparison(output_dir)
        
        # 2. Template comparison
        plots['template_comparison'] = self._plot_template_comparison(output_dir)
        
        # 3. Shot count comparison
        plots['shot_comparison'] = self._plot_shot_comparison(output_dir)
        
        # 4. Reranking comparison
        plots['rerank_comparison'] = self._plot_rerank_comparison(output_dir)
        
        # 5. Embedding source/model comparison
        plots['embedding_comparison'] = self._plot_embedding_comparison(output_dir)
        
        # 6. Interactions
        plots['interactions'] = self._plot_interactions(output_dir)
        
        # 7. Overall best configurations
        plots['best_configs'] = self._plot_best_configurations(output_dir)
        
        return plots
    
    def _plot_method_comparison(self, output_dir=None):
        """Plot comparison of retrieval methods"""
        method_impact = self._analyze_method_impact()
        
        # Create subplots for main metrics
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        axes = axes.flatten()
        
        metrics_to_plot = ['BLEU', 'COMET', 'ROUGE-L', 'CHRF++']
        
        for i, metric in enumerate(metrics_to_plot):
            sns.barplot(x='method', y=metric, data=method_impact, ax=axes[i])
            axes[i].set_title(f'Impact of Retrieval Method on {metric}')
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, horizontalalignment='right')
            # Add value labels on bars
            for p in axes[i].patches:
                axes[i].annotate(f"{p.get_height():.3f}", 
                             (p.get_x() + p.get_width() / 2., p.get_height()), 
                             ha = 'center', va = 'bottom',
                             xytext = (0, 5), textcoords = 'offset points')
        
        plt.tight_layout()
        
        if output_dir:
            fig.savefig(os.path.join(output_dir, 'method_comparison.png'))
            
        return fig
    
    def _plot_template_comparison(self, output_dir=None):
        """Plot comparison of prompt templates"""
        template_impact = self._analyze_template_impact()
        
        # Create subplots for main metrics
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        axes = axes.flatten()
        
        metrics_to_plot = ['BLEU', 'COMET', 'ROUGE-L', 'CHRF++']
        
        for i, metric in enumerate(metrics_to_plot):
            sns.barplot(x='template', y=metric, data=template_impact, ax=axes[i])
            axes[i].set_title(f'Impact of Prompt Template on {metric}')
            # Add value labels on bars
            for p in axes[i].patches:
                axes[i].annotate(f"{p.get_height():.3f}", 
                             (p.get_x() + p.get_width() / 2., p.get_height()), 
                             ha = 'center', va = 'bottom',
                             xytext = (0, 5), textcoords = 'offset points')
        
        plt.tight_layout()
        
        if output_dir:
            fig.savefig(os.path.join(output_dir, 'template_comparison.png'))
            
        return fig
    
    def _plot_shot_comparison(self, output_dir=None):
        """Plot comparison of shot counts"""
        shot_impact = self._analyze_shot_impact()
        
        # Create subplots for main metrics
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        axes = axes.flatten()
        
        metrics_to_plot = ['BLEU', 'COMET', 'ROUGE-L', 'CHRF++']
        
        for i, metric in enumerate(metrics_to_plot):
            sns.barplot(x='shots', y=metric, data=shot_impact, ax=axes[i])
            axes[i].set_title(f'Impact of Shot Count on {metric}')
            # Add value labels on bars
            for p in axes[i].patches:
                axes[i].annotate(f"{p.get_height():.3f}", 
                             (p.get_x() + p.get_width() / 2., p.get_height()), 
                             ha = 'center', va = 'bottom',
                             xytext = (0, 5), textcoords = 'offset points')
        
        plt.tight_layout()
        
        if output_dir:
            fig.savefig(os.path.join(output_dir, 'shot_comparison.png'))
            
        return fig
    
    def _plot_rerank_comparison(self, output_dir=None):
        """Plot comparison of raw vs reranked results"""
        rerank_impact = self._analyze_rerank_impact()
        
        # Create subplots for main metrics
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        axes = axes.flatten()
        
        metrics_to_plot = ['BLEU', 'COMET', 'ROUGE-L', 'CHRF++']
        
        for i, metric in enumerate(metrics_to_plot):
            sns.barplot(x='result_type', y=metric, data=rerank_impact, ax=axes[i])
            axes[i].set_title(f'Impact of Reranking on {metric}')
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, horizontalalignment='right')
            # Add value labels on bars
            for p in axes[i].patches:
                axes[i].annotate(f"{p.get_height():.3f}", 
                             (p.get_x() + p.get_width() / 2., p.get_height()), 
                             ha = 'center', va = 'bottom',
                             xytext = (0, 5), textcoords = 'offset points')
        
        plt.tight_layout()
        
        if output_dir:
            fig.savefig(os.path.join(output_dir, 'rerank_comparison.png'))
            
        return fig
    
    def _plot_embedding_comparison(self, output_dir=None):
        """Plot comparison of embedding sources and models"""
        embedder_impact = self._analyze_embedder_impact()
        
        # Create combined identifier for plotting
        embedder_impact['embedder'] = embedder_impact['embedding_source'] + ': ' + embedder_impact['embedding_model']
        
        # Create subplots for main metrics
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        axes = axes.flatten()
        
        metrics_to_plot = ['BLEU', 'COMET', 'ROUGE-L', 'CHRF++']
        
        for i, metric in enumerate(metrics_to_plot):
            sns.barplot(x='embedder', y=metric, data=embedder_impact, ax=axes[i])
            axes[i].set_title(f'Impact of Embedder on {metric}')
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, horizontalalignment='right')
            # Add value labels on bars
            for p in axes[i].patches:
                axes[i].annotate(f"{p.get_height():.3f}", 
                             (p.get_x() + p.get_width() / 2., p.get_height()), 
                             ha = 'center', va = 'bottom',
                             xytext = (0, 5), textcoords = 'offset points')
        
        plt.tight_layout()
        
        if output_dir:
            fig.savefig(os.path.join(output_dir, 'embedder_comparison.png'))
            
        return fig
    
    def _plot_interactions(self, output_dir=None):
        """Plot interactions between components"""
        interactions = self._analyze_interactions()
        
        # Plot Method x Template interaction for BLEU score
        fig1, ax1 = plt.subplots(figsize=(14, 8))
        pivot_data = interactions['method_template'].pivot(index='method', columns='template', values='BLEU')
        sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax1)
        ax1.set_title('Method x Template Interaction (BLEU Score)')
        
        # Plot Method x Shots interaction for BLEU score
        fig2, ax2 = plt.subplots(figsize=(14, 8))
        pivot_data = interactions['method_shots'].pivot(index='method', columns='shots', values='BLEU')
        sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax2)
        ax2.set_title('Method x Shot Count Interaction (BLEU Score)')
        
        # Plot Method x Result Type interaction for BLEU score
        fig3, ax3 = plt.subplots(figsize=(14, 8))
        pivot_data = interactions['method_result_type'].pivot(index='method', columns='result_type', values='BLEU')
        sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax3)
        ax3.set_title('Method x Result Type Interaction (BLEU Score)')
        
        # Plot Template x Shots interaction for BLEU score
        fig4, ax4 = plt.subplots(figsize=(14, 8))
        pivot_data = interactions['template_shots'].pivot(index='template', columns='shots', values='BLEU')
        sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax4)
        ax4.set_title('Template x Shot Count Interaction (BLEU Score)')
        
        # Plot Template x Result Type interaction for BLEU score
        fig5, ax5 = plt.subplots(figsize=(14, 8))
        pivot_data = interactions['template_result_type'].pivot(index='template', columns='result_type', values='BLEU')
        sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax5)
        ax5.set_title('Template x Result Type Interaction (BLEU Score)')
        
        if output_dir:
            fig1.savefig(os.path.join(output_dir, 'method_template_interaction.png'))
            fig2.savefig(os.path.join(output_dir, 'method_shots_interaction.png'))
            fig3.savefig(os.path.join(output_dir, 'method_result_type_interaction.png'))
            fig4.savefig(os.path.join(output_dir, 'template_shots_interaction.png'))
            fig5.savefig(os.path.join(output_dir, 'template_result_type_interaction.png'))
            
        return {
            'method_template': fig1,
            'method_shots': fig2,
            'method_result_type': fig3,
            'template_shots': fig4,
            'template_result_type': fig5
        }
    
    def _plot_best_configurations(self, output_dir=None):
        """Plot top configurations across metrics"""
        # For each metric, find the top 5 configurations
        top_configs = {}
        
        for metric in ['BLEU', 'COMET', 'ROUGE-L', 'CHRF++']:
            top_for_metric = self.data.sort_values(by=metric, ascending=False).head(10)
            
            # Create configuration label
            top_for_metric['config'] = top_for_metric.apply(
                lambda row: f"{row['method']} | {row['template']} | {row['shots']} shots", axis=1
            )
            
            # Get top 5 configs
            top_configs[metric] = top_for_metric[['config', metric]].head(5)
        
        # Plot the top configurations
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        axes = axes.flatten()
        
        metrics_to_plot = ['BLEU', 'COMET', 'ROUGE-L', 'CHRF++']
        
        for i, metric in enumerate(metrics_to_plot):
            sns.barplot(x=metric, y='config', data=top_configs[metric], ax=axes[i])
            axes[i].set_title(f'Top 5 Configurations by {metric}')
            # Add value labels on bars
            for p in axes[i].patches:
                axes[i].annotate(f"{p.get_width():.3f}", 
                             (p.get_width(), p.get_y() + p.get_height() / 2), 
                             ha = 'left', va = 'center',
                             xytext = (5, 0), textcoords = 'offset points')
        
        plt.tight_layout()
        
        if output_dir:
            fig.savefig(os.path.join(output_dir, 'best_configurations.png'))
            
        return fig
    
    def generate_report(self):
        """Generate a detailed analysis report"""
        if self.data is None:
            self.load_all_metrics()
            
        # Get component analyses
        analyses = self.analyze_by_component()
        
        # Find best configurations for each metric
        best_configs = {}
        for metric in ['BLEU', 'COMET', 'ROUGE-L', 'CHRF++']:
            best_configs[metric] = self.data.loc[self.data[metric].idxmax()]
        
        # Find best configurations by result type (raw vs reranked)
        best_raw = self.data[self.data['result_type'] == 'raw'].loc[self.data[self.data['result_type'] == 'raw']['BLEU'].idxmax()]
        best_reranked = self.data[self.data['result_type'] == 'rerank'].loc[self.data[self.data['result_type'] == 'rerank']['BLEU'].idxmax()]
        
        # Generate report text
        report = """
# Translation System Comparative Analysis

## Overview
This report provides a detailed analysis of the translation system's performance across different components:
- Retrieval methods (BM25, Dense, Hybrid, Random)
- Prompt templates (T1-T5)
- Shot counts (1, 3, 5 shots)
- Reranking impact (raw vs reranked)
- Embedding sources and models

## Key Findings

### Best Overall Configurations
"""
        # Add best configurations section
        report += f"""
#### Best Configuration by BLEU Score ({best_configs['BLEU']['BLEU']:.2f}):
- Retrieval Method: {best_configs['BLEU']['method']}
- Template: {best_configs['BLEU']['template']}
- Shot Count: {best_configs['BLEU']['shots']}
- Reranking: {'Yes' if best_configs['BLEU']['result_type'] == 'rerank' else 'No'}
- Embedding: {best_configs['BLEU']['embedding_source']} - {best_configs['BLEU']['embedding_model']}

#### Best Configuration by COMET Score ({best_configs['COMET']['COMET']:.2f}):
- Retrieval Method: {best_configs['COMET']['method']}
- Template: {best_configs['COMET']['template']}
- Shot Count: {best_configs['COMET']['shots']}
- Reranking: {'Yes' if best_configs['COMET']['result_type'] == 'rerank' else 'No'}
- Embedding: {best_configs['COMET']['embedding_source']} - {best_configs['COMET']['embedding_model']}

### Reranking Impact
"""
        # Add reranking impact analysis
        rerank_impact = analyses['rerank_impact']
        for result_type in rerank_impact['result_type'].unique():
            rerank_data = rerank_impact[rerank_impact['result_type'] == result_type]
            report += f"""
#### {result_type.capitalize()} Results:
- BLEU: {rerank_data['BLEU'].values[0]:.2f}
- COMET: {rerank_data['COMET'].values[0]:.2f}
- ROUGE-L: {rerank_data['ROUGE-L'].values[0]:.2f}
- CHRF++: {rerank_data['CHRF++'].values[0]:.2f}
"""
        
        report += f"""
#### Best Raw Configuration (BLEU: {best_raw['BLEU']:.2f}):
- Retrieval Method: {best_raw['method']}
- Template: {best_raw['template']}
- Shot Count: {best_raw['shots']}
- Embedding: {best_raw['embedding_source']} - {best_raw['embedding_model']}

#### Best Reranked Configuration (BLEU: {best_reranked['BLEU']:.2f}):
- Retrieval Method: {best_reranked['method']}
- Template: {best_reranked['template']}
- Shot Count: {best_reranked['shots']}
- Embedding: {best_reranked['embedding_source']} - {best_reranked['embedding_model']}

### Retrieval Method Performance
"""
        # Add method impact analysis
        method_impact = analyses['method_impact']
        for method in method_impact['method'].unique():
            method_data = method_impact[method_impact['method'] == method]
            report += f"""
#### {method}:
- BLEU: {method_data['BLEU'].values[0]:.2f}
- COMET: {method_data['COMET'].values[0]:.2f}
- ROUGE-L: {method_data['ROUGE-L'].values[0]:.2f}
- CHRF++: {method_data['CHRF++'].values[0]:.2f}
"""
        
        report += """
### Template Performance
"""
        # Add template impact analysis
        template_impact = analyses['template_impact']
        for template in template_impact['template'].unique():
            template_data = template_impact[template_impact['template'] == template]
            report += f"""
#### {template}:
- BLEU: {template_data['BLEU'].values[0]:.2f}
- COMET: {template_data['COMET'].values[0]:.2f}
- ROUGE-L: {template_data['ROUGE-L'].values[0]:.2f}
- CHRF++: {template_data['CHRF++'].values[0]:.2f}
"""
        
        report += """
### Shot Count Impact
"""
        # Add shot impact analysis
        shot_impact = analyses['shot_impact']
        for shot in shot_impact['shots'].unique():
            shot_data = shot_impact[shot_impact['shots'] == shot]
            report += f"""
#### {shot} Shots:
- BLEU: {shot_data['BLEU'].values[0]:.2f}
- COMET: {shot_data['COMET'].values[0]:.2f}
- ROUGE-L: {shot_data['ROUGE-L'].values[0]:.2f}
- CHRF++: {shot_data['CHRF++'].values[0]:.2f}
"""
        
        report += """
### Embedding Model Performance
"""
        # Add embedder impact analysis
        embedder_impact = analyses['embedder_impact']
        for _, row in embedder_impact.iterrows():
            report += f"""
#### {row['embedding_source']} - {row['embedding_model']}:
- BLEU: {row['BLEU']:.2f}
- COMET: {row['COMET']:.2f}
- ROUGE-L: {row['ROUGE-L']:.2f}
- CHRF++: {row['CHRF++']:.2f}
"""
        
        # Add interaction analysis
        interactions = analyses['interactions']
        report += """
## Component Interactions

### Method × Template Interactions
The following combinations showed the best performance:
"""
        # Top 3 method-template combinations for BLEU
        top_method_template = interactions['method_template'].sort_values('BLEU', ascending=False).head(3)
        for _, row in top_method_template.iterrows():
            report += f"- {row['method']} with {row['template']}: BLEU = {row['BLEU']:.2f}, COMET = {row['COMET']:.2f}\n"
            
        report += """
### Method × Shot Count Interactions
Optimal shot counts for each retrieval method:
"""
        # Top method-shot combinations for BLEU
        for method in interactions['method_shots']['method'].unique():
            method_data = interactions['method_shots'][interactions['method_shots']['method'] == method]
            best_shot = method_data.loc[method_data['BLEU'].idxmax()]
            report += f"- {method}: Best with {best_shot['shots']} shots (BLEU = {best_shot['BLEU']:.2f})\n"
            
        report += """
### Template × Shot Count Interactions
Best shot counts for each template:
"""
        # Top template-shot combinations for BLEU
        for template in interactions['template_shots']['template'].unique():
            template_data = interactions['template_shots'][interactions['template_shots']['template'] == template]
            best_shot = template_data.loc[template_data['BLEU'].idxmax()]
            report += f"- {template}: Best with {best_shot['shots']} shots (BLEU = {best_shot['BLEU']:.2f})\n"
        
        # Add conclusions and recommendations
        report += """
## Conclusions and Recommendations

Based on the analysis, here are the key recommendations for optimal translation performance:

1. **Reranking**: """
        if best_reranked['BLEU'] > best_raw['BLEU']:
            report += f"Reranking improves performance by {best_reranked['BLEU'] - best_raw['BLEU']:.2f} BLEU points. Recommended."
        else:
            report += "Reranking does not provide significant improvements. Not recommended."

        report += f"""
2. **Retrieval Method**: {best_configs['BLEU']['method']} performs best overall.
3. **Prompt Template**: {best_configs['BLEU']['template']} yields the best results.
4. **Shot Count**: {best_configs['BLEU']['shots']} shots provides optimal performance.
5. **Embedding Choice**: {best_configs['BLEU']['embedding_source']} - {best_configs['BLEU']['embedding_model']} is the most effective.

### Best Overall Configuration
The optimal configuration for translation is:
- Retrieval Method: {best_configs['BLEU']['method']}
- Template: {best_configs['BLEU']['template']}
- Shot Count: {best_configs['BLEU']['shots']}
- Reranking: {'Yes' if best_configs['BLEU']['result_type'] == 'rerank' else 'No'}
- Embedding: {best_configs['BLEU']['embedding_source']} - {best_configs['BLEU']['embedding_model']}

This configuration achieves:
- BLEU: {best_configs['BLEU']['BLEU']:.2f}
- COMET: {best_configs['BLEU']['COMET']:.2f}
- ROUGE-L: {best_configs['BLEU']['ROUGE-L']:.2f}
- CHRF++: {best_configs['BLEU']['CHRF++']:.2f}

### Further Exploration
- Investigate why certain method-template combinations perform better
- Test additional prompt templates
- Explore hybrid retrieval approaches
- Analyze the impact of different embedding models in more detail
"""
        
        return report

# Example usage
if __name__ == "__main__":
    base_dir = "/Users/aravadikesh/Documents/GitHub/NoteAid-translation-EngToSpa/retrieval_translation/experiment_results"
    analyzer = TranslationExperimentAnalyzer(base_dir)
    
    # Load all metrics data
    data = analyzer.load_all_metrics()
    print(f"Loaded {len(data)} experiment results")
    
    # Generate plots in output directory
    output_dir = "retrieval_translation/analysis_output"
    plots = analyzer.generate_plots(output_dir)
    
    # Generate report
    report = analyzer.generate_report()
    
    # Save report to file
    with open(os.path.join(output_dir, "analysis_report.md"), "w") as f:
        f.write(report)
    
    print(f"Analysis complete. Results saved to {output_dir}")
