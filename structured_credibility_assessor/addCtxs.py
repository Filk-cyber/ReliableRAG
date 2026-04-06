import json
import time
import concurrent.futures
from threading import Lock
from zhipuai import ZhipuAI
from retrievers.e5_mistral import get_e5_mistral_embeddings_for_query, get_e5_mistral_embeddings_for_document
import torch
from typing import List, Dict, Any, Tuple


class OptimizedTitleGenerator:
    def __init__(self, api_key: str, max_workers: int = 10):
        """
        Initialize optimized title generator

        Args:
            api_key: ZhipuAI API key
            max_workers: Maximum number of concurrent threads
        """
        self.client = ZhipuAI(api_key=api_key)
        self.max_workers = max_workers
        self.progress_lock = Lock()
        self.completed_count = 0
        self.total_count = 0

        # Default identifiers
        self.DEFAULT_TITLE = "DEFAULT_TITLE_PLACEHOLDER"
        self.EMPTY_TITLE = ""

        # Minimum configuration values
        self.MIN_WORKERS = 1

        # Keep original prompt template
        self.instruction = """Your task is to generate a single concise title for the given English paragraph. The generated title should be less than 10 words.
Here are 2 examples, you should follow the output format below:
##########
Passage:
Boston College (also referred to as BC) is a private Jesuit Catholic research university located in the affluent village of Chestnut Hill, Massachusetts, United States, 6 mi west of downtown Boston. It has 9,100 full-time undergraduates and almost 5,000 graduate students. The university's name reflects its early history as a liberal arts college and preparatory school (now Boston College High School) in Dorchester. It is a member of the 568 Group and the Association of Jesuit Colleges and Universities. Its main campus is a historic district and features some of the earliest examples of collegiate gothic architecture in North America.

Title: Boston College



Passage:
The Rideau River Residence Association (RRRA) is the student organization that represents undergraduate students living in residence at Carleton University. It was founded in 1968 as the Carleton University Residence Association. Following a protracted fight with the university in the mid-1970s, it was renamed in its present form. It is a non-profit corporation that serves as Canada's oldest and largest residence association. Its membership consists of roughly 3,600 undergraduate students enrolled at the university living in residence. With an annual budget of approximately $1.4 million and three executives alongside volunteer staff, RRRA serves as an advocate for residence students and provides a variety of services, events, and programs to its members.

Title: Rideau River Residence Association
##########
"""

        self.user_input_template = """Passage: {passage}
Title: 
"""

    def get_dataset_demonstrations(self, dataset):
        """Get dataset demonstration examples"""
        if dataset == "hotpotqa":
            from prompts import generate_knowledge_triples_hotpotqa_examplars
            demonstrations = generate_knowledge_triples_hotpotqa_examplars
        elif dataset == "2wikimultihopqa":
            from prompts import generate_knowledge_triples_2wikimultihopqa_examplars
            demonstrations = generate_knowledge_triples_2wikimultihopqa_examplars
        elif dataset == "musique":
            from prompts import generate_knowledge_triples_musique_examplars
            demonstrations = generate_knowledge_triples_musique_examplars
        else:
            raise ValueError(f"{dataset} is not a supported dataset!")
        return demonstrations

    def split_sentences(self, text):
        """Split sentences by period, keeping the period"""
        parts = text.split('.')
        sentences = []

        for i, part in enumerate(parts):
            part = part.strip()
            if part:
                if i < len(parts) - 1:
                    sentences.append(part + '.')
                else:
                    if text.endswith('.'):
                        sentences.append(part + '.')
                    else:
                        sentences.append(part)
        return sentences

    def generate_title_single(self, passage: str) -> str:
        """
        Generate title for a single passage

        Args:
            passage: Passage text

        Returns:
            Generated title
        """
        try:
            user_input = self.user_input_template.format(passage=passage)
            response = self.client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {"role": "system", "content": self.instruction},
                    {"role": "user", "content": user_input},
                ],
                stream=True,
            )

            full_response_content = ""
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_response_content += delta.content

            return full_response_content.strip()

        except Exception as e:
            print(f"Title generation failed: {str(e)}")
            return self.DEFAULT_TITLE

    def call_api_with_retry(self, passage: str, max_retries: int = 3) -> str:
        """
        Call API with retry mechanism

        Args:
            passage: Passage text
            max_retries: Maximum number of retries

        Returns:
            Generated title
        """
        for attempt in range(max_retries):
            try:
                result = self.generate_title_single(passage)
                if result != self.DEFAULT_TITLE and result.strip():
                    return result
                else:
                    print(f"Attempt {attempt + 1} got empty or default result")
            except Exception as e:
                print(f"API call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        print(f"API call finally failed, returning default value")
        return self.DEFAULT_TITLE

    def generate_titles_only(self, passage_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Single-threaded processing function for title generation only

        Args:
            passage_data: Dictionary containing passage information

        Returns:
            Result dictionary containing title
        """
        try:
            item_idx = passage_data['item_idx']
            paragraph_idx = passage_data['paragraph_idx']
            paragraph = passage_data['paragraph']

            # Generate title
            title = self.call_api_with_retry(paragraph)

            with self.progress_lock:
                self.completed_count += 1
                print(
                    f"Title generation progress: {self.completed_count}/{self.total_count} - Item {item_idx + 1}, Paragraph {paragraph_idx + 1}")

            return {
                'item_idx': item_idx,
                'paragraph_idx': paragraph_idx,
                'paragraph': paragraph,
                'title': title,
                'success': True
            }

        except Exception as e:
            print(f"Error processing item {item_idx}, paragraph {paragraph_idx}: {e}")
            return {
                'item_idx': item_idx,
                'paragraph_idx': paragraph_idx,
                'paragraph': passage_data['paragraph'],
                'title': self.DEFAULT_TITLE,
                'success': False
            }

    def generate_title_for_ctx(self, passage_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate title for a single ctx

        Args:
            passage_data: Dictionary containing passage information

        Returns:
            Result dictionary containing title
        """
        try:
            item_idx = passage_data['item_idx']
            ctx_idx = passage_data['ctx_idx']
            paragraph = passage_data['paragraph']

            # Generate title
            title = self.call_api_with_retry(paragraph)

            with self.progress_lock:
                self.completed_count += 1
                print(
                    f"Title generation progress: {self.completed_count}/{self.total_count} - Item {item_idx + 1}, ctx {ctx_idx + 1}")

            return {
                'item_idx': item_idx,
                'ctx_idx': ctx_idx,
                'paragraph': paragraph,
                'title': title,
                'success': True
            }

        except Exception as e:
            print(f"Error processing item {item_idx}, ctx {ctx_idx}: {e}")
            return {
                'item_idx': item_idx,
                'ctx_idx': ctx_idx,
                'paragraph': paragraph,
                'title': self.DEFAULT_TITLE,
                'success': False
            }

    def collect_all_paragraphs(self, dataset: List[Dict]) -> List[Dict[str, Any]]:
        """
        Collect all paragraph data that needs to be processed

        Args:
            dataset: Dataset

        Returns:
            List of all paragraph data
        """
        all_paragraphs_data = []

        for item_idx, item in enumerate(dataset):
            # Check if ori_fake field exists
            if 'ori_fake' in item and isinstance(item['ori_fake'], list):
                for paragraph_idx, paragraph in enumerate(item['ori_fake']):
                    if paragraph.strip():
                        all_paragraphs_data.append({
                            'item_idx': item_idx,
                            'paragraph_idx': paragraph_idx,
                            'paragraph': paragraph
                        })

        return all_paragraphs_data

    def collect_default_title_paragraphs(self, dataset: List[Dict]) -> List[Dict[str, Any]]:
        """
        Collect paragraph data with default or empty titles

        Args:
            dataset: Dataset

        Returns:
            List of paragraph data that needs title regeneration
        """
        paragraphs_data = []

        for item_idx, item in enumerate(dataset):
            if 'ctxs' in item and isinstance(item['ctxs'], list):
                for ctx_idx, ctx in enumerate(item['ctxs']):
                    if 'title' in ctx and 'text' in ctx:
                        title = ctx['title']
                        if title == self.DEFAULT_TITLE or title.strip() == self.EMPTY_TITLE:
                            paragraphs_data.append({
                                'item_idx': item_idx,
                                'ctx_idx': ctx_idx,
                                'paragraph': ctx['text']
                            })

        return paragraphs_data

    def collect_missing_title_paragraphs(self, dataset: List[Dict]) -> List[Dict[str, Any]]:
        """
        Collect paragraph data with missing title fields

        Args:
            dataset: Dataset

        Returns:
            List of paragraph data that needs title added
        """
        paragraphs_data = []

        for item_idx, item in enumerate(dataset):
            if 'ctxs' in item and isinstance(item['ctxs'], list):
                for ctx_idx, ctx in enumerate(item['ctxs']):
                    if 'title' not in ctx and 'text' in ctx:
                        paragraphs_data.append({
                            'item_idx': item_idx,
                            'ctx_idx': ctx_idx,
                            'paragraph': ctx['text']
                        })

        return paragraphs_data

    def stage1_generate_all_titles(self, dataset: List[Dict], output_file: str) -> List[Dict[str, Any]]:
        """
        Stage 1: Batch generate all titles

        Args:
            dataset: Dataset
            output_file: Output file path

        Returns:
            All title generation results
        """
        print("=" * 80)
        print("Stage 1: Batch generate all titles")
        print("=" * 80)

        # Collect all paragraphs
        all_paragraphs_data = self.collect_all_paragraphs(dataset)
        self.total_count = len(all_paragraphs_data)
        self.completed_count = 0

        print(f"Total {self.total_count} titles to generate")

        if self.total_count == 0:
            print("No paragraphs to process")
            return []

        # Use multi-threading to generate titles
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_paragraph = {
                executor.submit(self.generate_titles_only, paragraph_data): paragraph_data
                for paragraph_data in all_paragraphs_data
            }

            for future in concurrent.futures.as_completed(future_to_paragraph):
                result = future.result()
                results.append(result)

        success_count = sum(1 for r in results if r['success'])
        print(f"Title generation completed: {success_count}/{len(results)} succeeded")

        # Save title generation results
        self.save_titles_results(results, output_file, "stage1_titles")

        return results

    def stage1_generate_titles_for_ctxs(self, paragraphs_data: List[Dict[str, Any]], output_file: str) -> List[Dict[str, Any]]:
        """
        Stage 1: Generate titles for existing ctxs

        Args:
            paragraphs_data: List of paragraph data
            output_file: Output file path

        Returns:
            Title generation results
        """
        print("=" * 80)
        print("Stage 1: Generate titles for existing ctxs")
        print("=" * 80)

        self.total_count = len(paragraphs_data)
        self.completed_count = 0

        print(f"Total {self.total_count} titles to generate")

        if self.total_count == 0:
            print("No paragraphs to process")
            return []

        # Use multi-threading to generate titles
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_paragraph = {
                executor.submit(self.generate_title_for_ctx, paragraph_data): paragraph_data
                for paragraph_data in paragraphs_data
            }

            for future in concurrent.futures.as_completed(future_to_paragraph):
                result = future.result()
                results.append(result)

        success_count = sum(1 for r in results if r['success'])
        print(f"Title generation completed: {success_count}/{len(results)} succeeded")

        return results

    def save_titles_results(self, titles_results: List[Dict], output_file: str, stage: str):
        """
        Save title generation results

        Args:
            titles_results: List of title generation results
            output_file: Output file path
            stage: Stage identifier
        """
        try:
            temp_file = f"{output_file}.{stage}.json"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(titles_results, f, ensure_ascii=False, indent=2)
            print(f"{stage} stage title results saved to: {temp_file}")
        except Exception as e:
            print(f"Failed to save {stage} stage title results: {e}")

    def stage2_calculate_similarities(self, titles_results: List[Dict], dataset_name: str, output_file: str) -> List[Dict]:
        """
        Stage 2: Calculate all similarities

        Args:
            titles_results: Title generation results
            dataset_name: Dataset name
            output_file: Output file path

        Returns:
            Results with similarities
        """
        print("=" * 80)
        print("Stage 2: Calculate all similarities")
        print("=" * 80)

        if not titles_results:
            print("No title results to process")
            return []

        # Get demonstration embeddings
        dataset_demonstrations = self.get_dataset_demonstrations(dataset_name)
        demonstration_texts = ["title: {} text: {}".format(demo["title"], demo["text"]) for demo in
                               dataset_demonstrations]

        print(f"Calculating demonstration embeddings...")
        demonstration_embeddings = get_e5_mistral_embeddings_for_document(
            doc_list=demonstration_texts,
            max_length=256,
            batch_size=4,
        )

        # Build document text list
        document_texts = []
        for result in titles_results:
            if result['success']:
                document_text = f"title: {result['title']} text: {result['paragraph']}"
                document_texts.append(document_text)
            else:
                document_texts.append("")  # Placeholder

        print(f"Calculating embeddings for {len(document_texts)} documents...")

        # Process in batches to avoid memory overflow
        batch_size = 13  # Can be adjusted based on GPU memory
        all_similarities = []

        for i in range(0, len(document_texts), batch_size):
            batch_texts = document_texts[i:i + batch_size]
            # Filter out empty texts
            valid_texts = [text for text in batch_texts if text.strip()]

            if valid_texts:
                print(f"Processing batch {i // batch_size + 1}/{(len(document_texts) + batch_size - 1) // batch_size}")

                # Calculate embeddings for current batch
                documents_embeddings = get_e5_mistral_embeddings_for_query(
                    "retrieve_semantically_similar_text",
                    query_list=valid_texts,
                    max_length=256,
                    batch_size=4,
                )

                # Calculate similarities
                similarities = torch.matmul(documents_embeddings, demonstration_embeddings.T)
                demonstration_ranks = torch.argsort(similarities, dim=1, descending=True)

                # Add to total results, handling empty texts
                valid_idx = 0
                for j, text in enumerate(batch_texts):
                    if text.strip():
                        all_similarities.append(demonstration_ranks[valid_idx].tolist())
                        valid_idx += 1
                    else:
                        all_similarities.append([])  # Placeholder for empty text

                # Clear GPU memory
                del documents_embeddings, similarities, demonstration_ranks
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Add similarity results to titles_results
        enhanced_results = []
        for i, result in enumerate(titles_results):
            enhanced_result = result.copy()
            if i < len(all_similarities):
                enhanced_result['ranked_prompt_indices'] = all_similarities[i]
            else:
                enhanced_result['ranked_prompt_indices'] = []
            enhanced_results.append(enhanced_result)

        print(f"Similarity calculation completed")

        # Save results with similarities
        self.save_titles_results(enhanced_results, output_file, "stage2_similarities")

        return enhanced_results

    def stage3_apply_to_dataset(self, enhanced_results: List[Dict], dataset: List[Dict]) -> List[Dict]:
        """
        Stage 3: Apply results to dataset

        Args:
            enhanced_results: Results with titles and similarities
            dataset: Original dataset

        Returns:
            Updated dataset
        """
        print("=" * 80)
        print("Stage 3: Apply results to dataset")
        print("=" * 80)

        if not enhanced_results:
            print("No results to apply")
            return dataset

        # Group results by item_idx
        results_by_item = {}
        for result in enhanced_results:
            item_idx = result['item_idx']
            if item_idx not in results_by_item:
                results_by_item[item_idx] = []
            results_by_item[item_idx].append(result)

        # Apply results to dataset
        processed_items = 0
        for item_idx, item_results in results_by_item.items():
            if item_idx < len(dataset):
                existing_ctxs = dataset[item_idx].get('ctxs', [])

                # Create ctx object for each paragraph
                for result in item_results:
                    if result['success'] and result.get('ranked_prompt_indices'):
                        # Generate unique ID
                        new_id = len(existing_ctxs)

                        # Split sentences
                        sentences = self.split_sentences(result['paragraph'])

                        # Construct new ctx object
                        new_ctx = {
                            "id": str(new_id),
                            "title": result['title'],
                            "text": result['paragraph'],
                            "sentences": sentences,
                            "ranked_prompt_indices": result['ranked_prompt_indices']
                        }

                        existing_ctxs.append(new_ctx)

                # Update dataset
                dataset[item_idx]['ctxs'] = existing_ctxs
                processed_items += 1

        print(f"Dataset update completed, processed {processed_items} items")
        return dataset

    def stage3_apply_to_existing_ctxs(self, enhanced_results: List[Dict], dataset: List[Dict]) -> List[Dict]:
        """
        Stage 3: Apply titles and similarity results to existing ctxs

        Args:
            enhanced_results: Results with titles and similarities
            dataset: Dataset

        Returns:
            Updated dataset
        """
        print("=" * 80)
        print("Stage 3: Apply titles and similarities to existing ctxs")
        print("=" * 80)

        processed_count = 0
        for result in enhanced_results:
            if result['success'] and result.get('ranked_prompt_indices'):
                item_idx = result['item_idx']
                ctx_idx = result['ctx_idx']
                title = result['title']
                ranked_prompt_indices = result['ranked_prompt_indices']

                try:
                    if item_idx < len(dataset) and 'ctxs' in dataset[item_idx]:
                        ctxs = dataset[item_idx]['ctxs']
                        if ctx_idx < len(ctxs):
                            ctxs[ctx_idx]['title'] = title
                            ctxs[ctx_idx]['ranked_prompt_indices'] = ranked_prompt_indices
                            processed_count += 1
                except (IndexError, KeyError) as e:
                    print(f"Error applying result to item {item_idx}, ctx {ctx_idx}: {e}")

        print(f"Title and similarity application completed, processed {processed_count} ctxs")
        return dataset

    def check_default_or_empty_titles(self, dataset: List[Dict]) -> List[Dict]:
        """
        Check if there are default or empty titles in the dataset and extract them

        Args:
            dataset: Dataset

        Returns:
            Item data with default or empty values
        """
        failed_items = []

        for item_idx, item in enumerate(dataset):
            if 'ctxs' in item and isinstance(item['ctxs'], list):
                has_default_or_empty = False
                for ctx in item['ctxs']:
                    if 'title' in ctx:
                        title = ctx['title']
                        if title == self.DEFAULT_TITLE or title.strip() == self.EMPTY_TITLE:
                            has_default_or_empty = True
                            break

                if has_default_or_empty:
                    failed_items.append({
                        'item_idx': item_idx,
                        'item': item,
                        'dataset': 'hotpotqa'  # Default dataset
                    })

        return failed_items

    def count_default_or_empty_titles(self, dataset: List[Dict]) -> Tuple[int, int]:
        """
        Count the number of default or empty values

        Args:
            dataset: Dataset

        Returns:
            (items_with_issues, total_titles_with_issues): Number of items and titles with issues
        """
        items_with_issues = 0
        total_titles_with_issues = 0

        for item in dataset:
            if 'ctxs' in item and isinstance(item['ctxs'], list):
                item_has_issues = False
                for ctx in item['ctxs']:
                    if 'title' in ctx:
                        title = ctx['title']
                        if title == self.DEFAULT_TITLE or title.strip() == self.EMPTY_TITLE:
                            total_titles_with_issues += 1
                            item_has_issues = True

                if item_has_issues:
                    items_with_issues += 1

        return items_with_issues, total_titles_with_issues

    def check_missing_title_fields(self, dataset: List[Dict]) -> List[Dict]:
        """
        Check if there are ctxs with missing title fields in the dataset and extract them

        Args:
            dataset: Dataset

        Returns:
            Item data with missing title fields
        """
        missing_items = []

        for item_idx, item in enumerate(dataset):
            if 'ctxs' in item and isinstance(item['ctxs'], list):
                has_missing_title = False
                for ctx in item['ctxs']:
                    if 'title' not in ctx:
                        has_missing_title = True
                        break

                if has_missing_title:
                    missing_items.append({
                        'item_idx': item_idx,
                        'item': item,
                        'dataset': 'hotpotqa'  # Default dataset
                    })

        return missing_items

    def count_missing_title_fields(self, dataset: List[Dict]) -> int:
        """
        Count the number of missing title fields

        Args:
            dataset: Dataset

        Returns:
            Number of ctxs with missing title fields
        """
        missing_count = 0

        for item in dataset:
            if 'ctxs' in item and isinstance(item['ctxs'], list):
                for ctx in item['ctxs']:
                    if 'title' not in ctx:
                        missing_count += 1

        return missing_count

    def process_default_title_check(self, input_file: str, output_file: str, dataset_name: str = "hotpotqa"):
        """
        Execute default title check and repair (complete three-stage processing)

        Args:
            input_file: Input file path
            output_file: Output file path
            dataset_name: Dataset name
        """
        print("🔍 Executing default title check mode (complete three-stage processing)")
        print("=" * 80)

        # Read input file
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
        except Exception as e:
            print(f"Failed to read input file: {e}")
            return

        if not isinstance(dataset, list):
            dataset = [dataset]

        # Count default titles
        items_count, titles_count = self.count_default_or_empty_titles(dataset)
        print(f"Found default or empty titles: {items_count} items, {titles_count} titles")

        if titles_count == 0:
            print("✅ No default or empty titles found, no processing needed")
            return

        # Stage 1: Collect paragraphs that need title regeneration and generate titles
        paragraphs_data = self.collect_default_title_paragraphs(dataset)
        print(f"Collected {len(paragraphs_data)} paragraphs that need title regeneration")

        titles_results = self.stage1_generate_titles_for_ctxs(paragraphs_data, output_file)

        # Stage 2: Calculate similarities
        enhanced_results = self.stage2_calculate_similarities(titles_results, dataset_name, output_file)

        # Stage 3: Apply to dataset
        updated_dataset = self.stage3_apply_to_existing_ctxs(enhanced_results, dataset)

        # Save results
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(updated_dataset, f, ensure_ascii=False, indent=2)
            print(f"✅ Default title check completed! Results saved to: {output_file}")

            # Clean up temporary files
            import os
            for stage in ["stage1_titles", "stage2_similarities"]:
                temp_file = f"{output_file}.{stage}.json"
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"Cleaned up temporary file: {temp_file}")

            # Final statistics
            final_items_count, final_titles_count = self.count_default_or_empty_titles(updated_dataset)
            print(f"🏁 Post-processing statistics:")
            print(f"   - Remaining default/empty titles: {final_items_count} items, {final_titles_count} titles")
            print(f"   - Successfully repaired: {titles_count - final_titles_count} titles")

        except Exception as e:
            print(f"Failed to save final output file: {e}")

    def process_missing_title_check(self, input_file: str, output_file: str, dataset_name: str = "hotpotqa"):
        """
        Execute missing title check and repair (complete three-stage processing)

        Args:
            input_file: Input file path
            output_file: Output file path
            dataset_name: Dataset name
        """
        print("🔍 Executing missing title check mode (complete three-stage processing)")
        print("=" * 80)

        # Read input file
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
        except Exception as e:
            print(f"Failed to read input file: {e}")
            return

        if not isinstance(dataset, list):
            dataset = [dataset]

        # Count missing titles
        missing_count = self.count_missing_title_fields(dataset)
        print(f"Found missing title fields: {missing_count} ctxs")

        if missing_count == 0:
            print("✅ No missing title fields found, no processing needed")
            return

        # Stage 1: Collect paragraphs that need title added and generate titles
        paragraphs_data = self.collect_missing_title_paragraphs(dataset)
        print(f"Collected {len(paragraphs_data)} paragraphs that need title added")

        titles_results = self.stage1_generate_titles_for_ctxs(paragraphs_data, output_file)

        # Stage 2: Calculate similarities
        enhanced_results = self.stage2_calculate_similarities(titles_results, dataset_name, output_file)

        # Stage 3: Apply to dataset
        updated_dataset = self.stage3_apply_to_existing_ctxs(enhanced_results, dataset)

        # Save results
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(updated_dataset, f, ensure_ascii=False, indent=2)
            print(f"✅ Missing title check completed! Results saved to: {output_file}")

            # Clean up temporary files
            import os
            for stage in ["stage1_titles", "stage2_similarities"]:
                temp_file = f"{output_file}.{stage}.json"
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"Cleaned up temporary file: {temp_file}")

            # Final statistics
            final_missing_count = self.count_missing_title_fields(updated_dataset)
            print(f"🏁 Post-processing statistics:")
            print(f"   - Remaining missing title fields: {final_missing_count} ctxs")
            print(f"   - Successfully added: {missing_count - final_missing_count} titles")

        except Exception as e:
            print(f"Failed to save final output file: {e}")

    def process_dataset_optimized_separated(self, input_file: str, output_file: str, dataset_name: str = "hotpotqa"):
        """
        Complete three-stage dataset processing

        Args:
            input_file: Input file path
            output_file: Output file path
            dataset_name: Dataset name
        """
        print(f"Starting complete three-stage processing for dataset: {input_file}")
        print("🚀 Executing complete three-stage processing")
        print("   Stage 1: Batch generate all titles")
        print("   Stage 2: Batch calculate all similarities")
        print("   Stage 3: Apply results to dataset")

        # Read input file
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
        except Exception as e:
            print(f"Failed to read input file: {e}")
            return

        if not isinstance(dataset, list):
            dataset = [dataset]

        # Execute Stage 1: Generate titles
        titles_results = self.stage1_generate_all_titles(dataset, output_file)

        # Execute Stage 2: Calculate similarities
        enhanced_results = self.stage2_calculate_similarities(titles_results, dataset_name, output_file)

        # Execute Stage 3: Apply to dataset
        updated_dataset = self.stage3_apply_to_dataset(enhanced_results, dataset)

        # Save final results
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(updated_dataset, f, ensure_ascii=False, indent=2)
            print(f"✅ Complete processing finished! Results saved to: {output_file}")

            # Clean up temporary files
            import os
            for stage in ["stage1_titles", "stage2_similarities"]:
                temp_file = f"{output_file}.{stage}.json"
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"Cleaned up temporary file: {temp_file}")

            # Final statistics
            default_items_count, default_titles_count = self.count_default_or_empty_titles(updated_dataset)
            missing_count = self.count_missing_title_fields(updated_dataset)
            print(f"🏁 Final statistics:")
            print(f"   - Remaining problematic items: {default_items_count}, problematic titles: {default_titles_count}")
            print(f"   - Remaining missing title fields: {missing_count}")

        except Exception as e:
            print(f"Failed to save final output file: {e}")

    def save_progress(self, dataset: List[Dict], output_file: str, stage: str):
        """
        Save intermediate progress
        """
        try:
            temp_file = f"{output_file}.{stage}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            print(f"{stage} stage progress saved to temporary file: {temp_file}")
        except Exception as e:
            print(f"Failed to save {stage} stage progress: {e}")


def main():
    """
    Main function - Usage example
    """
    # Configuration parameters
    API_KEY = ""  # Please fill in your ZhipuAI API Key
    INPUT_FILE = "wiki_test1000_add_orifake.json"
    OUTPUT_FILE = "wiki_test1000_add_ctxs.json"

    # Parallel processing parameters
    MAX_WORKERS = 3000  # Number of concurrent threads, adjust based on API limits
    DATASET_NAME = "2wikimultihopqa"  # Dataset name

    # ⭐⭐ Check mode control parameters
    CHECK_DEFAULT_TITLES = False  # Set to True to execute default title check mode
    CHECK_MISSING_TITLES = False  # Set to True to execute missing title check mode

    # Parameter description:
    # 1. CHECK_DEFAULT_TITLES=True: Check and repair default or empty titles (complete three-stage processing)
    # 2. CHECK_MISSING_TITLES=True: Check and add missing title fields (complete three-stage processing)
    # 3. All check parameters set to False: Execute complete three-stage processing

    if not API_KEY:
        print("Error: Please set your ZhipuAI API Key first")
        return

    # Create generator instance
    generator = OptimizedTitleGenerator(API_KEY, max_workers=MAX_WORKERS)

    # Decide execution mode based on parameters
    if CHECK_DEFAULT_TITLES:
        print("🔍 Enable default title check mode: Check and repair default or empty titles (complete three-stage processing)")
        generator.process_default_title_check(INPUT_FILE, OUTPUT_FILE, DATASET_NAME)
    elif CHECK_MISSING_TITLES:
        print("🔍 Enable missing title check mode: Check and add missing title fields (complete three-stage processing)")
        generator.process_missing_title_check(INPUT_FILE, OUTPUT_FILE, DATASET_NAME)
    else:
        print("🚀 Enable complete three-stage processing mode")
        generator.process_dataset_optimized_separated(INPUT_FILE, OUTPUT_FILE, DATASET_NAME)

    print(f"📂 Input file: {INPUT_FILE}")
    print(f"📂 Output file: {OUTPUT_FILE}")

    print("\n" + "=" * 80)
    print("Processing completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
