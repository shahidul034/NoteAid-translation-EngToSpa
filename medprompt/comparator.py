import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rouge_score import rouge_scorer
import os

class TranslationComparator:
    def __init__(self, datapath1: str, datapath2: str, output_dir: str = 'translation_analysis'):
        """
        Initialize the comparator with two JSON files containing translations
        
        Args:
        - datapath1 (str): Path to first translation JSON
        - datapath2 (str): Path to second translation JSON
        - output_dir (str): Directory to save output files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        
        # Load both datasets
        self.data1 = pd.read_json(datapath1)
        self.data2 = pd.read_json(datapath2)
        
        # Ensure both datasets have the same length and order
        assert len(self.data1) == len(self.data2), "Datasets must have the same number of entries"
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def compute_detailed_metrics(self) -> pd.DataFrame:
        """
        Compute detailed metrics for each translation pair
        
        Returns:
        - DataFrame with detailed comparison metrics
        """
        # Prepare comparison dataframe
        comparison_df = pd.DataFrame({
            'original_english': self.data1['original_english'],
            'target_spanish': self.data1['target_spanish'],
            'translation1': self.data1['translated_spanish'],
            'translation2': self.data2['translated_spanish']
        })
        
        # Compute metrics
        rouge_metrics = []
        length_differences = []
        word_overlap = []
        
        for i in range(len(comparison_df)):
            # ROUGE Scores
            rouge_scores = self.rouge_scorer.score(
                comparison_df.loc[i, 'target_spanish'], 
                comparison_df.loc[i, 'translation1']
            )
            rouge_scores2 = self.rouge_scorer.score(
                comparison_df.loc[i, 'target_spanish'], 
                comparison_df.loc[i, 'translation2']
            )
            
            # Length differences
            len_trans1 = len(comparison_df.loc[i, 'translation1'].split())
            len_trans2 = len(comparison_df.loc[i, 'translation2'].split())
            len_target = len(comparison_df.loc[i, 'target_spanish'].split())
            
            # Word overlap
            words_trans1 = set(comparison_df.loc[i, 'translation1'].split())
            words_trans2 = set(comparison_df.loc[i, 'translation2'].split())
            words_target = set(comparison_df.loc[i, 'target_spanish'].split())
            
            rouge_metrics.append({
                'rouge1_f1_1': rouge_scores['rouge1'].fmeasure,
                'rouge1_f1_2': rouge_scores2['rouge1'].fmeasure,
                'rouge2_f1_1': rouge_scores['rouge2'].fmeasure,
                'rouge2_f1_2': rouge_scores2['rouge2'].fmeasure,
                'rougeL_f1_1': rouge_scores['rougeL'].fmeasure,
                'rougeL_f1_2': rouge_scores2['rougeL'].fmeasure
            })
            
            length_differences.append({
                'len_trans1': len_trans1,
                'len_trans2': len_trans2,
                'len_target': len_target,
                'len_diff_abs': abs(len_trans1 - len_trans2),
                'len_diff_pct1': abs(len_trans1 - len_target) / len_target * 100,
                'len_diff_pct2': abs(len_trans2 - len_target) / len_target * 100
            })
            
            word_overlap.append({
                'overlap_with_target1': len(words_trans1 & words_target) / len(words_target),
                'overlap_with_target2': len(words_trans2 & words_target) / len(words_target),
                'overlap_between_translations': len(words_trans1 & words_trans2) / len(words_target)
            })
        
        # Add metrics to dataframe
        comparison_df = pd.concat([
            comparison_df, 
            pd.DataFrame(rouge_metrics),
            pd.DataFrame(length_differences),
            pd.DataFrame(word_overlap)
        ], axis=1)
        
        # Compute divergence score
        comparison_df['divergence_score'] = (
            (1 - comparison_df['rouge1_f1_1']) + 
            (1 - comparison_df['rouge2_f1_1']) + 
            abs(comparison_df['len_diff_abs']) / 10 + 
            (1 - comparison_df['overlap_with_target1'])
        )
        
        # Sort by divergence score
        return comparison_df.sort_values('divergence_score', ascending=False)
    
    def visualize_divergence(self, detailed_metrics: pd.DataFrame):
        """
        Create visualizations of translation differences
        
        Args:
        - detailed_metrics (pd.DataFrame): Detailed metrics DataFrame
        """
        # 1. Divergence Score Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(detailed_metrics['divergence_score'], kde=True)
        plt.title('Distribution of Translation Divergence Scores')
        plt.xlabel('Divergence Score')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'divergence_distribution.png'))
        plt.close()
        
        # 2. ROUGE Score Comparison
        plt.figure(figsize=(12, 6))
        plt.scatter(detailed_metrics['rouge1_f1_1'], detailed_metrics['rouge1_f1_2'], alpha=0.5)
        plt.title('ROUGE-1 F1 Score Comparison')
        plt.xlabel('Translation 1 ROUGE-1 F1')
        plt.ylabel('Translation 2 ROUGE-1 F1')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'rouge_comparison.png'))
        plt.close()
        
        # 3. Length Differences
        plt.figure(figsize=(10, 6))
        plt.scatter(detailed_metrics['len_trans1'], detailed_metrics['len_trans2'], alpha=0.5)
        plt.title('Translation Length Comparison')
        plt.xlabel('Translation 1 Length')
        plt.ylabel('Translation 2 Length')
        plt.plot([0, max(detailed_metrics['len_trans1'].max(), detailed_metrics['len_trans2'].max())], 
                 [0, max(detailed_metrics['len_trans1'].max(), detailed_metrics['len_trans2'].max())], 
                 color='red', linestyle='--')  # Diagonal line
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'length_comparison.png'))
        plt.close()
    
    def save_detailed_results(self, detailed_metrics: pd.DataFrame):
        """
        Save detailed results to a text file
        
        Args:
        - detailed_metrics (pd.DataFrame): Detailed metrics DataFrame
        """
        # Save full detailed results
        detailed_metrics.to_csv(os.path.join(self.output_dir, 'full_translation_metrics.csv'), index=False)
        
        # Create a more readable text log
        with open(os.path.join(self.output_dir, 'translation_analysis_log.txt'), 'w', encoding='utf-8') as f:
            f.write("Translation Comparison Analysis\n")
            f.write("=" * 50 + "\n\n")
            
            # Top 20 most divergent translations
            f.write("Top 20 Most Divergent Translations\n")
            f.write("-" * 30 + "\n")
            top_divergent = detailed_metrics.head(20)
            
            for idx, row in top_divergent.iterrows():
                f.write(f"Divergence Score: {row['divergence_score']:.4f}\n")
                f.write(f"Original English: {row['original_english']}\n")
                f.write(f"Target Spanish:   {row['target_spanish']}\n")
                f.write(f"Translation 1:    {row['translation1']}\n")
                f.write(f"Translation 2:    {row['translation2']}\n")
                f.write("\nMetrics:\n")
                f.write(f"ROUGE-1 F1 (1st Translation): {row['rouge1_f1_1']:.4f}\n")
                f.write(f"ROUGE-1 F1 (2nd Translation): {row['rouge1_f1_2']:.4f}\n")
                f.write(f"Length Translation 1: {row['len_trans1']}\n")
                f.write(f"Length Translation 2: {row['len_trans2']}\n")
                f.write(f"Length Target: {row['len_target']}\n")
                f.write("\n" + "=" * 50 + "\n\n")
            
            # Overall statistics
            f.write("\nOverall Translation Metrics\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Translations: {len(detailed_metrics)}\n")
            f.write(f"Average Divergence Score: {detailed_metrics['divergence_score'].mean():.4f}\n")
            f.write(f"Median Divergence Score: {detailed_metrics['divergence_score'].median():.4f}\n")
            f.write(f"Max Divergence Score: {detailed_metrics['divergence_score'].max():.4f}\n")
            f.write(f"Min Divergence Score: {detailed_metrics['divergence_score'].min():.4f}\n")
    
    def full_analysis(self):
        """
        Perform full translation comparison analysis
        """
        # Compute detailed metrics
        detailed_metrics = self.compute_detailed_metrics()
        
        # Create visualizations
        self.visualize_divergence(detailed_metrics)
        
        # Save detailed results
        self.save_detailed_results(detailed_metrics)
        
        print(f"Analysis complete. Results saved in {self.output_dir}")
        return detailed_metrics

# Example usage
comparator = TranslationComparator('/Users/aravadikesh/Documents/GitHub/NoteAid-translation-EngToSpa/medprompt/results/llama8b/translated_output.json', '/Users/aravadikesh/Documents/GitHub/NoteAid-translation-EngToSpa/maps/results/llama8b/translated_output.json')
results = comparator.full_analysis()
