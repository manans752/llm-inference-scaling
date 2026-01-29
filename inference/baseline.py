"""
Single-pass baseline decoding implementation.
Uses Hugging Face Inference Client for greedy decoding with fixed max tokens.
"""

import json
import os
from typing import Dict, List, Optional
import time
from datetime import datetime

try:
    from huggingface_hub import InferenceClient
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False
    print("Warning: huggingface_hub not installed. Install with: pip install huggingface_hub --break-system-packages")


class BaselineDecoder:
    """Handles single-pass greedy decoding using Hugging Face Inference Client."""

    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.3", api_token: Optional[str] = None):
        """
        Initialize the decoder.

        Args:
            model_name: HuggingFace model identifier
                Recommended models (as of Jan 2025):
                - mistralai/Mistral-7B-Instruct-v0.3 (good general purpose, 7B params)
                - google/gemma-2-2b-it (smaller, faster, 2B params)
                - Qwen/Qwen2.5-7B-Instruct (good alternative, 7B params)
            api_token: HuggingFace API token (or set HF_TOKEN env variable)
        """
        if not HAS_HF_HUB:
            raise ImportError("Please install huggingface_hub: pip install huggingface_hub --break-system-packages")

        self.model_name = model_name
        self.api_token = api_token or os.getenv("HF_TOKEN")

        if not self.api_token:
            raise ValueError("No API token provided. Set HF_TOKEN environment variable or pass api_token parameter.")

        # Initialize InferenceClient
        self.client = InferenceClient(token=self.api_token)
        print(f"Initialized InferenceClient with model: {model_name}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.01,  # Very low for greedy-like behavior (0.0 causes issues)
        retry_attempts: int = 3,
        retry_delay: int = 5
    ) -> Dict:
        """
        Generate a response using greedy decoding.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.01 = nearly greedy, 0.0 causes API errors)
            retry_attempts: Number of retry attempts if API fails
            retry_delay: Delay between retries in seconds

        Returns:
            Dictionary containing generated text and metadata
        """
        for attempt in range(retry_attempts):
            try:
                # Use text_generation with the InferenceClient
                response = self.client.text_generation(
                    prompt,
                    model=self.model_name,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=False,  # Greedy decoding
                    return_full_text=False  # Only return generated text
                )

                # Response is just a string with the generated text
                generated_text = response.strip()

                return {
                    "generated_text": generated_text,
                    "prompt": prompt,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature
                }

            except Exception as e:
                error_msg = str(e)
                print(f"Attempt {attempt + 1}/{retry_attempts} failed: {error_msg}")

                # Handle model loading
                if "loading" in error_msg.lower():
                    print(f"Model is loading. Waiting {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue

                if attempt < retry_attempts - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise

        raise Exception("Max retry attempts reached")

    def run_baseline_inference(
        self,
        tasks_path: str = "data/tasks.json",
        prompt_template_path: str = "prompts/baseline.txt",
        output_path: str = "results/baseline_results.json",
        max_new_tokens: int = 100
    ) -> List[Dict]:
        """
        Run baseline inference on all tasks.

        Args:
            tasks_path: Path to tasks JSON file
            prompt_template_path: Path to baseline prompt template
            output_path: Path to save results
            max_new_tokens: Maximum tokens to generate per question

        Returns:
            List of results dictionaries
        """
        # Load tasks
        with open(tasks_path, 'r') as f:
            tasks = json.load(f)

        # Load prompt template
        with open(prompt_template_path, 'r') as f:
            prompt_template = f.read().strip()

        results = []

        print(f"Running baseline inference on {len(tasks)} tasks...")
        print(f"Model: {self.model_name}")
        print(f"Max tokens: {max_new_tokens}")
        print("-" * 80)

        for i, task in enumerate(tasks):
            print(f"\nTask {i+1}/{len(tasks)}: {task['id']}")
            print(f"Question: {task['question']}")

            # Format prompt
            prompt = prompt_template.replace("{question}", task['question'])

            # Generate response
            start_time = time.time()
            generation = self.generate(prompt, max_new_tokens=max_new_tokens)
            end_time = time.time()

            # Extract answer from generated text
            generated_text = generation['generated_text'].strip()

            # Parse the answer (extract final number or text)
            extracted_answer = self._extract_answer(generated_text)

            # Check correctness
            is_correct = self._check_answer(extracted_answer, task['answer'])

            # Count tokens (rough estimate: ~4 chars per token)
            token_count = len(generated_text) // 4

            result = {
                "task_id": task['id'],
                "question": task['question'],
                "expected_answer": task['answer'],
                "generated_text": generated_text,
                "extracted_answer": extracted_answer,
                "is_correct": is_correct,
                "token_count": token_count,
                "inference_time_seconds": round(end_time - start_time, 2),
                "max_new_tokens": max_new_tokens,
                "timestamp": datetime.now().isoformat()
            }

            results.append(result)

            print(f"Generated: {generated_text[:100]}...")
            print(f"Extracted answer: {extracted_answer}")
            print(f"Expected answer: {task['answer']}")
            print(f"Correct: {is_correct}")
            print(f"Tokens: {token_count}")

        # Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Print summary
        self._print_summary(results)

        return results

    def _extract_answer(self, text: str) -> str:
        """
        Extract the final answer from generated text.
        Uses simple heuristics to find the answer.
        """
        text = text.strip()

        # Look for common answer patterns
        patterns = [
            "answer is",
            "answer:",
            "the answer is",
            "therefore",
            "thus",
            "=",
        ]

        text_lower = text.lower()

        # Try to find answer after these patterns
        for pattern in patterns:
            if pattern in text_lower:
                idx = text_lower.rfind(pattern)
                potential_answer = text[idx + len(pattern):].strip()
                # Get first line or sentence
                potential_answer = potential_answer.split('\n')[0].split('.')[0].strip()
                if potential_answer:
                    # Clean up
                    potential_answer = potential_answer.strip('.,!?:; ')
                    return potential_answer

        # If no pattern found, take last line
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines:
            return lines[-1].strip('.,!?:; ')

        return text

    def _check_answer(self, extracted: str, expected) -> bool:
        """
        Check if extracted answer matches expected answer.
        Handles both numeric and text answers.
        """
        # Normalize strings
        extracted_clean = str(extracted).strip().lower()
        expected_clean = str(expected).strip().lower()

        # Direct match
        if extracted_clean == expected_clean:
            return True

        # Try numeric comparison
        try:
            extracted_num = float(extracted_clean)
            expected_num = float(expected_clean)
            # Allow small floating point differences
            return abs(extracted_num - expected_num) < 0.01
        except (ValueError, TypeError):
            pass

        # Check if expected answer is contained in extracted
        if expected_clean in extracted_clean:
            return True

        return False

    def _print_summary(self, results: List[Dict]):
        """Print summary statistics."""
        total = len(results)
        correct = sum(1 for r in results if r['is_correct'])
        accuracy = (correct / total * 100) if total > 0 else 0
        avg_tokens = sum(r['token_count'] for r in results) / total if total > 0 else 0
        avg_time = sum(r['inference_time_seconds'] for r in results) / total if total > 0 else 0

        print("\n" + "=" * 80)
        print("BASELINE INFERENCE SUMMARY")
        print("=" * 80)
        print(f"Total tasks: {total}")
        print(f"Correct: {correct}")
        print(f"Incorrect: {total - correct}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Average tokens: {avg_tokens:.1f}")
        print(f"Average time per task: {avg_time:.2f}s")
        print("=" * 80)


def main():
    """Main function to run baseline inference."""
    # Initialize decoder
    # Using Mistral-7B-Instruct-v0.3 (available and works well)
    # Other options: "google/gemma-2-2b-it" or "Qwen/Qwen2.5-7B-Instruct"
    decoder = BaselineDecoder(
        model_name="Qwen/Qwen2.5-7B-Instruct"
        # api_token will be read from HF_TOKEN environment variable
    )

    # Run baseline inference
    results = decoder.run_baseline_inference(
        tasks_path="data/tasks.json",
        prompt_template_path="prompts/baseline.txt",
        output_path="results/baseline_results.json",
        max_new_tokens=100
    )


if __name__ == "__main__":
    main()

# # import json
# # import os
# # import requests
# # from pathlib import Path
# #
# #
# # # -----------------------------
# # # CONFIG
# # # -----------------------------
# #
# # MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
# # API_URL = "https://router.huggingface.co/v1/completions"
# # TASKS_FILE = "tasks.json"
# # MAX_NEW_TOKENS = 50
# #
# # HF_TOKEN = os.getenv("HF_TOKEN")
# # if HF_TOKEN is None:
# #     raise RuntimeError("HF_TOKEN environment variable not set")
# #
# # HEADERS = {
# #     "Authorization": f"Bearer {HF_TOKEN}",
# #     "Content-Type": "application/json"
# # }
# #
# # # -----------------------------
# # # PROMPT FORMATTING
# # # -----------------------------
# #
# # def format_prompt(question: str) -> str:
# #     # Instruct models still accept plain text
# #     return f"Question: {question}\nAnswer:"
# #
# # # -----------------------------
# # # HF ROUTER CALL
# # # -----------------------------
# #
# # def call_hf_inference(prompt: str, max_new_tokens: int) -> str:
# #     payload = {
# #         "model": MODEL_NAME,
# #         "prompt": prompt,
# #         "max_tokens": max_new_tokens,
# #         "temperature": 0.0,   # greedy decoding
# #         "top_p": 1.0,
# #     }
# #
# #     response = requests.post(
# #         API_URL,
# #         headers=HEADERS,
# #         json=payload,
# #         timeout=60
# #     )
# #
# #     if response.status_code != 200:
# #         raise RuntimeError(
# #             f"HF API error {response.status_code}: {response.text}"
# #         )
# #
# #     data = response.json()
# #     return data["choices"][0]["text"]
# #
# # # -----------------------------
# # # LOAD TASKS
# # # -----------------------------
# #
# # HERE = Path(__file__).parent
# # PATH_TO_TASKS = HERE.parent / "data" / "tasks.json"
# #
# # with PATH_TO_TASKS.open("r", encoding="utf-8") as f:
# #     tasks = json.load(f)
# #
# # print(f"Confirm loaded {len(tasks)} tasks.")
# #
# # # -----------------------------
# # # RUN BASELINE
# # # -----------------------------
# #
# # results = []
# #
# # for idx, task in enumerate(tasks):
# #     question = task["question"]
# #     gold_answer = task["answer"]
# #
# #     prompt = format_prompt(question)
# #     output = call_hf_inference(prompt, MAX_NEW_TOKENS)
# #
# #     is_correct = gold_answer.strip() in output.strip()
# #
# #     results.append({
# #         "id": idx,
# #         "question": question,
# #         "gold_answer": gold_answer,
# #         "model_output": output,
# #         "token_budget": MAX_NEW_TOKENS,
# #         "correct": is_correct
# #     })
# #
# #     print(f"[{idx + 1}/{len(tasks)}] correct={is_correct}")
# #
# # # -----------------------------
# # # SAVE RESULTS
# # # -----------------------------
# #
# # os.makedirs("results", exist_ok=True)
# #
# # with open("results/baseline_results.json", "w") as f:
# #     json.dump(results, f, indent=2)
# #
# # print("Baseline run complete.")
# #
# # # import os
# # # # import requests
# # # import json
# # # import pandas as pd
# # # import re
# # # from pathlib import Path
# # # from huggingface_hub import InferenceClient
# # #
# # # MAX_NEW_TOKENS = 50
# # # HF_TOKEN = os.getenv("HF_TOKEN")
# # # # client = InferenceClient(token=HF_TOKEN)
# # # API_URL = "https://router.huggingface.co/models/mistralai/Ministral-3-3B-Base-2512"
# # # MODEL_NAME = "mistralai/Ministral-3-3B-Base-2512"
# # #
# # # # headers = {"Authorization": f"Bearer {HF_TOKEN}",
# # # #            "Content-Type": "application/json"}
# # #
# # # client = InferenceClient(
# # #     token=HF_TOKEN
# # # )
# # #
# # # HERE = Path(__file__).parent
# # # PATH_TO_TASKS = HERE.parent / "data" / "tasks.json"
# # #
# # # with PATH_TO_TASKS.open("r", encoding="utf-8") as f:
# # #     tasks = json.load(f)
# # #
# # # print(f"Confirm loaded {len(tasks)} tasks.")
# # #
# # # # def query(prompt):
# # # #     response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
# # # #     return response.json()
# # #
# # # # wraps a user question in a specific template for Mistral model inference
# # # def format_prompt(question: str) -> str:
# # #     return f"<s>[INST] {question} [/INST]"
# # #
# # # # def call_hf_api(prompt: str, max_new_tokens: int = 50) -> str:
# # # #     payload = {
# # # #         "inputs": prompt,
# # # #         "parameters": {"max_new_tokens":max_new_tokens, "do_sample": False}
# # # #     }
# # # #     response = requests.post(API_URL, headers=headers, json=payload)
# # # #     if response.status_code != 200:
# # # #         raise RuntimeError(f"API request failed with status code {response.status_code}")
# # # #     result = response.json()
# # # #     if isinstance(result, list) and "generated_text" in result[0]:
# # # #         return result[0]["generated_text"]
# # # #     else:
# # # #         raise ValueError(f"Unexpected API response: {result}")
# # #
# # # # def call_hf_inference(prompt: str, max_new_tokens: int = 50) -> str:
# # # #     """Send prompt to HF InferenceClient for single-pass greedy decoding"""
# # # #     response = client.text_generation(
# # # #         prompt,
# # # #         model=MODEL_ID,
# # # #         max_new_tokens=max_new_tokens,
# # # #         do_sample=False
# # # #     )
# # # #     # response is a dict containing 'generated_text'
# # # #     return response.generated_text
# # #
# # # # def call_hf_inference(given_prompt, max_new_tokens=50):
# # # #     response = client.chat_completion(
# # # #         messages=[
# # # #             {"role": "user", "content": given_prompt}
# # # #         ],
# # # #         max_tokens=max_new_tokens,
# # # #         temperature=0.7,
# # # #     )
# # # #
# # # #     return response.choices[0].message.content
# # #
# # # def call_hf_inference(client: InferenceClient, prompt: str, max_new_tokens: int) -> str:
# # #     response = client.post(
# # #         url=API_URL,
# # #         json={
# # #             "model": MODEL_NAME,
# # #             "prompt": prompt,
# # #             "max_tokens": max_new_tokens,
# # #             "temperature": 0.0,   # greedy decoding
# # #             "top_p": 1.0,
# # #         },
# # #     )
# # #
# # #     # HF router uses OpenAI-style schema
# # #     return response["choices"][0]["text"]
# # #
# # #
# # #
# # #
# # # def extract_answer(text: str):
# # #     numbers = re.findall(r"\d+", text)
# # #     return int(numbers[-1]) if numbers else None
# # #
# # # #single-pass decoding
# # # results = []
# # #
# # # for idx, task in enumerate(tasks):
# # #     question = task["question"]
# # #     gold_answer = task["answer"]
# # #
# # #     prompt = format_prompt(question)
# # #     output = call_hf_inference(client, prompt, MAX_NEW_TOKENS)
# # #
# # #     is_correct = gold_answer.strip() in output.strip()
# # #
# # #     results.append({
# # #         "id": idx,
# # #         "question": question,
# # #         "gold_answer": gold_answer,
# # #         "model_output": output,
# # #         "token_budget": MAX_NEW_TOKENS,
# # #         "correct": is_correct
# # #     })
# # #
# # #     print(f"[{idx+1}/{len(tasks)}] correct={is_correct}")
# # #
# # # df = pd.DataFrame(results)
# # # df.to_csv("baseline_api_results.csv", index=False)
# # #
# # # accuracy = df["correct"].mean()
# # # print(f"Single-pass greedy decoding accuracy on {len(tasks)} tasks: {accuracy:.2f}")
# # # print("Results saved to baseline_api_results.csv")
#
# import json
# import os
# from huggingface_hub import InferenceClient
# from typing import Dict, List, Any
# import re
#
#
# class BaselineDecoder:
#     def __init__(self, model_name: str, hf_token: str, max_new_tokens: int = 100):
#         """
#         Initialize the baseline decoder with greedy decoding.
#
#         Args:
#             model_name: HuggingFace model identifier (e.g., 'meta-llama/Llama-3-8B-Instruct')
#             hf_token: Your HuggingFace API token
#             max_new_tokens: Maximum tokens to generate (default: 100)
#         """
#         self.model_name = model_name
#         self.max_new_tokens = max_new_tokens
#         self.client = InferenceClient(token=hf_token)
#
#     # def generate(self, prompt: str) -> Dict[str, Any]:
#     #     """
#     #     Generate a single response using greedy decoding.
#     #
#     #     Args:
#     #         prompt: Input prompt text
#     #
#     #     Returns:
#     #         Dictionary containing:
#     #             - generated_text: The full generated response
#     #             - tokens_generated: Number of tokens in the response
#     #     """
#     #     # Greedy decoding parameters:
#     #     # - do_sample=False ensures greedy (deterministic) decoding
#     #     # - max_new_tokens limits output length
#     #     # - temperature=0.0 is redundant with do_sample=False but emphasizes determinism
#     #     response = self.client.text_generation(
#     #         prompt=prompt,
#     #         model=self.model_name,
#     #         max_new_tokens=self.max_new_tokens,
#     #         do_sample=False,  # Greedy decoding (always pick highest probability token)
#     #         temperature=1.0,  # Not used when do_sample=False
#     #         details=True,  # Get token-level details
#     #         return_full_text=False  # Only return generated text, not the prompt
#     #     )
#     #
#     #     # Extract information from response
#     #     if hasattr(response, 'generated_text'):
#     #         generated_text = response.generated_text
#     #         tokens_generated = response.details.generated_tokens
#     #     else:
#     #         # Fallback if details not available
#     #         generated_text = response
#     #         tokens_generated = self._estimate_tokens(generated_text)
#     #
#     #     return {
#     #         "generated_text": generated_text,
#     #         "tokens_generated": tokens_generated
#     #     }
#     def generate(self, prompt: str) -> Dict[str, Any]:
#         """
#         Generate a single response using greedy decoding via chat completion.
#
#         Args:
#             prompt: Input prompt text
#
#         Returns:
#             Dictionary containing:
#                 - generated_text: The full generated response
#                 - tokens_generated: Number of tokens in the response
#         """
#         # Format prompt as chat messages
#         messages = [
#             {"role": "user", "content": prompt}
#         ]
#
#         # Greedy decoding parameters for chat completion:
#         # - temperature=0.0 ensures deterministic (greedy) decoding
#         # - max_tokens limits output length
#         response = self.client.chat_completion(
#             messages=messages,
#             model=self.model_name,
#             max_tokens=self.max_new_tokens,
#             temperature=0.0,  # Greedy decoding (always pick highest probability)
#             top_p=1.0,  # No nucleus sampling
#         )
#
#         # Extract the generated text
#         generated_text = response.choices[0].message.content
#
#         # Count tokens (estimate if exact count not available)
#         # The usage field contains token counts if available
#         if hasattr(response, 'usage') and response.usage:
#             tokens_generated = response.usage.completion_tokens
#         else:
#             tokens_generated = self._estimate_tokens(generated_text)
#
#         return {
#             "generated_text": generated_text,
#             "tokens_generated": tokens_generated
#         }
#     def _estimate_tokens(self, text: str) -> int:
#         """
#         Rough token estimation if exact count not available.
#         Approximation: ~4 characters per token for English text.
#         """
#         return len(text) // 4
#
#
# def extract_answer(generated_text: str) -> str:
#     """
#     Extract the final answer from generated text.
#     Looks for numeric answers or specific text patterns.
#
#     Args:
#         generated_text: The model's generated response
#
#     Returns:
#         Extracted answer as a string
#     """
#     # Clean up the text
#     text = generated_text.strip()
#
#     # Try to find the last number in the response (for arithmetic questions)
#     numbers = re.findall(r'-?\d+\.?\d*', text)
#     if numbers:
#         return numbers[-1]  # Return the last number found
#
#     # For text answers, return the last line or sentence
#     lines = [line.strip() for line in text.split('\n') if line.strip()]
#     if lines:
#         return lines[-1]
#
#     return text
#
#
# def check_correctness(predicted: str, expected: Any) -> bool:
#     """
#     Check if the predicted answer matches the expected answer.
#
#     Args:
#         predicted: Model's predicted answer
#         expected: Ground truth answer
#
#     Returns:
#         True if correct, False otherwise
#     """
#     # Convert expected to string for comparison
#     expected_str = str(expected)
#
#     # Try direct string match (case-insensitive)
#     if predicted.lower().strip() == expected_str.lower().strip():
#         return True
#
#     # Try numeric comparison with tolerance
#     try:
#         pred_num = float(predicted)
#         exp_num = float(expected)
#         # Allow small floating point differences
#         return abs(pred_num - exp_num) < 0.01
#     except (ValueError, TypeError):
#         pass
#
#     # Check if expected answer is contained in prediction
#     if expected_str.lower() in predicted.lower():
#         return True
#
#     return False
#
#
# def run_baseline_inference(
#         tasks_path: str,
#         model_name: str,
#         hf_token: str,
#         output_path: str,
#         max_new_tokens: int = 100
# ):
#     """
#     Run baseline inference on all tasks and save results.
#
#     Args:
#         tasks_path: Path to tasks.json file
#         model_name: HuggingFace model identifier
#         hf_token: HuggingFace API token
#         output_path: Path to save results JSON
#         max_new_tokens: Maximum tokens to generate per question
#     """
#     # Load tasks
#     with open(tasks_path, 'r') as f:
#         tasks = json.load(f)
#
#     # Initialize decoder
#     decoder = BaselineDecoder(
#         model_name=model_name,
#         hf_token=hf_token,
#         max_new_tokens=max_new_tokens
#     )
#
#     # Store results
#     results = []
#     correct_count = 0
#     total_tokens = 0
#
#     print(f"Running baseline inference on {len(tasks)} tasks...")
#     print(f"Model: {model_name}")
#     print(f"Max tokens: {max_new_tokens}")
#     print("-" * 60)
#
#     # Process each task
#     for i, task in enumerate(tasks, 1):
#         question_id = task['id']
#         question = task['question']
#         expected_answer = task['answer']
#
#         # Create simple prompt
#         prompt = f"Question: {question}\nAnswer:"
#
#         # Generate response
#         try:
#             output = decoder.generate(prompt)
#             generated_text = output['generated_text']
#             tokens_used = output['tokens_generated']
#
#             # Extract answer
#             predicted_answer = extract_answer(generated_text)
#
#             # Check correctness
#             is_correct = check_correctness(predicted_answer, expected_answer)
#
#             if is_correct:
#                 correct_count += 1
#
#             total_tokens += tokens_used
#
#             # Store result
#             result = {
#                 "id": question_id,
#                 "question": question,
#                 "expected_answer": expected_answer,
#                 "generated_text": generated_text,
#                 "predicted_answer": predicted_answer,
#                 "tokens_used": tokens_used,
#                 "correct": is_correct
#             }
#             results.append(result)
#
#             # Print progress
#             print(f"[{i}/{len(tasks)}] {question_id}: {'✓' if is_correct else '✗'} "
#                   f"(tokens: {tokens_used})")
#
#         except Exception as e:
#             print(f"[{i}/{len(tasks)}] {question_id}: ERROR - {str(e)}")
#             results.append({
#                 "id": question_id,
#                 "question": question,
#                 "expected_answer": expected_answer,
#                 "error": str(e),
#                 "correct": False
#             })
#
#     # Calculate statistics
#     accuracy = correct_count / len(tasks) if tasks else 0
#     avg_tokens = total_tokens / len(tasks) if tasks else 0
#
#     # Prepare final output
#     output_data = {
#         "config": {
#             "model": model_name,
#             "max_new_tokens": max_new_tokens,
#             "decoding_strategy": "greedy",
#             "do_sample": False
#         },
#         "statistics": {
#             "total_questions": len(tasks),
#             "correct": correct_count,
#             "incorrect": len(tasks) - correct_count,
#             "accuracy": accuracy,
#             "total_tokens": total_tokens,
#             "avg_tokens_per_question": avg_tokens
#         },
#         "results": results
#     }
#
#     # Save results
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     with open(output_path, 'w') as f:
#         json.dump(output_data, f, indent=2)
#
#     # Print summary
#     print("-" * 60)
#     print(f"Baseline Inference Complete!")
#     print(f"Accuracy: {accuracy:.2%} ({correct_count}/{len(tasks)})")
#     print(f"Avg tokens per question: {avg_tokens:.1f}")
#     print(f"Total tokens used: {total_tokens}")
#     print(f"Results saved to: {output_path}")
#
#     return output_data
#
#
# if __name__ == "__main__":
#     # Configuration
#     TASKS_PATH = "data/tasks.json"
#     MODEL_NAME = "mistralai/Ministral-3-8B-Base-2512"  # or "mistralai/Mistral-7B-Instruct-v0.1"
#     HF_TOKEN = os.environ.get("HF_TOKEN")  # Read from environment variable
#     OUTPUT_PATH = "results/baseline_results.json"
#     MAX_NEW_TOKENS = 100
#
#     if not HF_TOKEN:
#         raise ValueError("Please set HF_TOKEN environment variable")
#
#     # Run baseline inference
#     run_baseline_inference(
#         tasks_path=TASKS_PATH,
#         model_name=MODEL_NAME,
#         hf_token=HF_TOKEN,
#         output_path=OUTPUT_PATH,
#         max_new_tokens=MAX_NEW_TOKENS
#     )
