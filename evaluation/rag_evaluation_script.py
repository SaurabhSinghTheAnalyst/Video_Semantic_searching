#!/usr/bin/env python3
"""
RAG Evaluation Script - With vs Without Reranker

This script evaluates the quality of our RAG system by testing specific questions
against the video transcripts with predefined expected answers. It compares
performance with and without cross-encoder reranking.
"""

import os
import json
import time
from typing import List, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from difflib import SequenceMatcher

from video_index_manager import VideoIndexManager
from video_search_engine import VideoSearchEngine

@dataclass
class TestCase:
    """Represents a single test case with question and expected answers"""
    question: str
    expected_answers: List[str]
    expected_videos: List[str]
    category: str
    difficulty: str  # "easy", "medium", "hard"
    description: str

@dataclass
class EvaluationResult:
    """Results from evaluating a single test case"""
    test_case: TestCase
    semantic_only_results: List[Dict[str, Any]]
    reranked_results: List[Dict[str, Any]]
    semantic_score: float
    reranked_score: float
    time_semantic: float
    time_reranked: float
    improvement: float

class RAGEvaluator:
    """Evaluates RAG system performance with comprehensive test cases"""
    
    def __init__(self):
        self.manager = VideoIndexManager()
        self.index = None
        self.test_cases = self._create_test_cases()
        
    def _create_test_cases(self) -> List[TestCase]:
        """Create comprehensive test cases based on transcript content"""
        return [
            # Easy Questions - Direct factual information
            TestCase(
                question="What is contextual retrieval?",
                expected_answers=[
                    "new chunking strategy from Anthropic",
                    "adds pre processing step",
                    "improve failure rates by 35 percent",
                    "adds contextual information to chunks"
                ],
                expected_videos=["Contextual Retrieval with Any LLM_ A Step-by-Step Guide.mp4"],
                category="Technical Concepts",
                difficulty="easy",
                description="Basic definition of contextual retrieval"
            ),
            
            TestCase(
                question="How much does contextual retrieval improve failure rates?",
                expected_answers=[
                    "35 percent",
                    "thirty five percent",
                    "when combined with PM25, 49 percent",
                    "forty nine percent"
                ],
                expected_videos=["Contextual Retrieval with Any LLM_ A Step-by-Step Guide.mp4"],
                category="Technical Metrics",
                difficulty="easy",
                description="Specific performance metrics"
            ),
            
            TestCase(
                question="What models can be used for contextual retrieval?",
                expected_answers=[
                    "Cloud three Haiku",
                    "OpenAI models",
                    "128,000 tokens context window",
                    "Gemini has 2,000,000 tokens",
                    "Lama 3.2 has 128,000 tokens"
                ],
                expected_videos=["Contextual Retrieval with Any LLM_ A Step-by-Step Guide.mp4"],
                category="Technical Implementation",
                difficulty="medium",
                description="Compatible models and requirements"
            ),
            
            # Medium Questions - Conceptual understanding
            TestCase(
                question="Why should you avoid passionate industries according to the business advice?",
                expected_answers=[
                    "passionate people enter",
                    "should not go",
                    "restaurant business",
                    "movie business",
                    "compete with passionate people",
                    "boring business higher likelihood money"
                ],
                expected_videos=["3.mp4"],
                category="Business Strategy",
                difficulty="medium",
                description="Investment philosophy about passionate vs boring industries"
            ),
            
            TestCase(
                question="What is the difference between status games and wealth creation?",
                expected_answers=[
                    "status games limited",
                    "wealth creation positive sum",
                    "status is zero sum",
                    "wealth can go infinitely",
                    "hunter gatherer times no wealth",
                    "ranking ladder hierarchy"
                ],
                expected_videos=["1.mp4"],
                category="Philosophy",
                difficulty="medium",
                description="Fundamental concepts about status vs wealth"
            ),
            
            TestCase(
                question="What industry would you invest 10 crore in for guaranteed profit?",
                expected_answers=[
                    "energy transition",
                    "fossil to nuclear renewable",
                    "electric vehicle company",
                    "battery company",
                    "solar farm",
                    "government incentives",
                    "boring not sexy"
                ],
                expected_videos=["3.mp4"],
                category="Investment Strategy",
                difficulty="medium",
                description="Specific investment recommendation"
            ),
            
            # Hard Questions - Complex reasoning and connections
            TestCase(
                question="How does the contextual retrieval implementation process work step by step?",
                expected_answers=[
                    "take whole document create chunks",
                    "feed chunks individually to prompt",
                    "situate within whole document",
                    "add 50 to 100 tokens",
                    "original document plus chunk",
                    "contextualized chunks embedding process"
                ],
                expected_videos=["Contextual Retrieval with Any LLM_ A Step-by-Step Guide.mp4"],
                category="Technical Process",
                difficulty="hard",
                description="Complete implementation workflow"
            ),
            
            TestCase(
                question="Why is trajectory more important than position in status games?",
                expected_answers=[
                    "Jimmy Carr idea",
                    "number 101 but last year 200",
                    "number two but last year number one",
                    "deceleration very tangible",
                    "evolution bleeding eventually dies",
                    "hardwired not to lose"
                ],
                expected_videos=["1.mp4"],
                category="Psychology",
                difficulty="hard",
                description="Complex concept about momentum vs absolute position"
            ),
            
            # Cross-video questions - Testing broader understanding
            TestCase(
                question="What are the common themes about building successful systems?",
                expected_answers=[
                    "avoid passionate competition",
                    "focus on wealth creation",
                    "systematic approach",
                    "long context models",
                    "embedding process",
                    "efficiency over emotion"
                ],
                expected_videos=["1.mp4", "3.mp4", "Contextual Retrieval with Any LLM_ A Step-by-Step Guide.mp4"],
                category="Cross-Domain Insights",
                difficulty="hard",
                description="Connections across different topics"
            ),
            
            # Edge cases and specific details
            TestCase(
                question="What specific prompt customization was shown for Tesla financial analysis?",
                expected_answers=[
                    "AI assistant specializing financial analysis",
                    "Tesla Q3 2023 financial report",
                    "identify main financial topics metrics",
                    "mention relevant time periods comparisons",
                    "Tesla overall financial health strategy",
                    "do not use phrases this chunks discusses"
                ],
                expected_videos=["Contextual Retrieval with Any LLM_ A Step-by-Step Guide.mp4"],
                category="Implementation Details",
                difficulty="hard",
                description="Specific prompt engineering example"
            )
        ]
    
    def setup_index(self):
        """Initialize the search index"""
        print("🔄 Loading search index...")
        self.index = self.manager.get_or_create_index('transcripts')
        print("✅ Index loaded successfully")
    
    def calculate_relevance_score(self, results: List[Dict[str, Any]], expected_answers: List[str], expected_videos: List[str]) -> float:
        """Calculate relevance score based on expected answers and videos"""
        if not results:
            return 0.0
        
        # Score components
        content_score = 0.0
        video_score = 0.0
        position_score = 0.0
        
        # Check content relevance in top results
        top_texts = [result['text'].lower() for result in results[:5]]
        combined_text = ' '.join(top_texts)
        
        matches = 0
        for expected in expected_answers:
            if any(expected.lower() in text for text in top_texts):
                matches += 1
        
        content_score = matches / len(expected_answers) if expected_answers else 0
        
        # Check video relevance
        result_videos = [result['video_file'] for result in results[:3]]
        video_matches = sum(1 for video in expected_videos if video in result_videos)
        video_score = video_matches / len(expected_videos) if expected_videos else 0
        
        # Position-based scoring (higher weight for top results)
        position_weights = [1.0, 0.8, 0.6, 0.4, 0.2]
        for i, result in enumerate(results[:5]):
            text = result['text'].lower()
            for expected in expected_answers:
                if expected.lower() in text:
                    position_score += position_weights[i] / len(expected_answers)
                    break
        
        # Combined score with weights
        final_score = (content_score * 0.5) + (video_score * 0.3) + (position_score * 0.2)
        return min(final_score, 1.0)
    
    def evaluate_test_case(self, test_case: TestCase) -> EvaluationResult:
        """Evaluate a single test case with both methods"""
        print(f"\n🔍 Testing: {test_case.question}")
        print(f"   Category: {test_case.category} | Difficulty: {test_case.difficulty}")
        
        # Test without reranker
        search_engine_semantic = VideoSearchEngine(self.index, use_reranker=False)
        start_time = time.time()
        semantic_results = search_engine_semantic.semantic_search(test_case.question, top_k=5)
        time_semantic = time.time() - start_time
        
        # Test with reranker
        search_engine_rerank = VideoSearchEngine(self.index, use_reranker=True)
        start_time = time.time()
        reranked_results = search_engine_rerank.semantic_search(test_case.question, top_k=5)
        time_reranked = time.time() - start_time
        
        # Calculate scores
        semantic_score = self.calculate_relevance_score(semantic_results, test_case.expected_answers, test_case.expected_videos)
        reranked_score = self.calculate_relevance_score(reranked_results, test_case.expected_answers, test_case.expected_videos)
        
        improvement = reranked_score - semantic_score
        
        print(f"   📊 Semantic Score: {semantic_score:.3f} | Reranked Score: {reranked_score:.3f} | Improvement: {improvement:+.3f}")
        
        return EvaluationResult(
            test_case=test_case,
            semantic_only_results=semantic_results,
            reranked_results=reranked_results,
            semantic_score=semantic_score,
            reranked_score=reranked_score,
            time_semantic=time_semantic,
            time_reranked=time_reranked,
            improvement=improvement
        )
    
    def run_evaluation(self, save_results: bool = True) -> List[EvaluationResult]:
        """Run complete evaluation suite"""
        print("🚀 COMPREHENSIVE RAG EVALUATION")
        print("=" * 60)
        
        if self.index is None:
            self.setup_index()
        
        results = []
        
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n[Test {i}/{len(self.test_cases)}]")
            try:
                result = self.evaluate_test_case(test_case)
                results.append(result)
            except Exception as e:
                print(f"❌ Error evaluating test case: {e}")
        
        # Generate summary
        self._print_summary(results)
        
        if save_results:
            self._save_results(results)
        
        return results
    
    def _print_summary(self, results: List[EvaluationResult]):
        """Print comprehensive evaluation summary"""
        print(f"\n🎯 EVALUATION SUMMARY")
        print("=" * 50)
        
        if not results:
            print("❌ No results to summarize")
            return
        
        # Overall metrics
        avg_semantic = sum(r.semantic_score for r in results) / len(results)
        avg_reranked = sum(r.reranked_score for r in results) / len(results)
        avg_improvement = sum(r.improvement for r in results) / len(results)
        avg_time_semantic = sum(r.time_semantic for r in results) / len(results)
        avg_time_reranked = sum(r.time_reranked for r in results) / len(results)
        
        print(f"📊 Overall Performance:")
        print(f"   Semantic Only:  {avg_semantic:.3f}")
        print(f"   With Reranker:  {avg_reranked:.3f}")
        print(f"   Improvement:    {avg_improvement:+.3f} ({avg_improvement/avg_semantic*100:+.1f}%)")
        print(f"   Time Overhead:  {avg_time_reranked - avg_time_semantic:.3f}s")
        
        # Performance by category
        categories = {}
        for result in results:
            category = result.test_case.category
            if category not in categories:
                categories[category] = []
            categories[category].append(result)
        
        print(f"\n📈 Performance by Category:")
        for category, cat_results in categories.items():
            cat_semantic = sum(r.semantic_score for r in cat_results) / len(cat_results)
            cat_reranked = sum(r.reranked_score for r in cat_results) / len(cat_results)
            cat_improvement = cat_reranked - cat_semantic
            print(f"   {category:20s}: {cat_semantic:.3f} → {cat_reranked:.3f} ({cat_improvement:+.3f})")
        
        # Performance by difficulty
        difficulties = {}
        for result in results:
            difficulty = result.test_case.difficulty
            if difficulty not in difficulties:
                difficulties[difficulty] = []
            difficulties[difficulty].append(result)
        
        print(f"\n🎚️  Performance by Difficulty:")
        for difficulty in ["easy", "medium", "hard"]:
            if difficulty in difficulties:
                diff_results = difficulties[difficulty]
                diff_semantic = sum(r.semantic_score for r in diff_results) / len(diff_results)
                diff_reranked = sum(r.reranked_score for r in diff_results) / len(diff_results)
                diff_improvement = diff_reranked - diff_semantic
                print(f"   {difficulty.title():8s}: {diff_semantic:.3f} → {diff_reranked:.3f} ({diff_improvement:+.3f})")
        
        # Best improvements
        best_improvements = sorted(results, key=lambda x: x.improvement, reverse=True)[:3]
        print(f"\n🏆 Top Improvements:")
        for i, result in enumerate(best_improvements, 1):
            print(f"   {i}. {result.test_case.question[:50]}...")
            print(f"      {result.semantic_score:.3f} → {result.reranked_score:.3f} ({result.improvement:+.3f})")
        
        # Cases where reranker helped least
        worst_improvements = sorted(results, key=lambda x: x.improvement)[:3]
        print(f"\n🔍 Needs Improvement:")
        for i, result in enumerate(worst_improvements, 1):
            print(f"   {i}. {result.test_case.question[:50]}...")
            print(f"      {result.semantic_score:.3f} → {result.reranked_score:.3f} ({result.improvement:+.3f})")
    
    def _save_results(self, results: List[EvaluationResult]):
        """Save detailed results to JSON file"""
        output_data = {
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_test_cases": len(results),
            "summary": {
                "avg_semantic_score": sum(r.semantic_score for r in results) / len(results),
                "avg_reranked_score": sum(r.reranked_score for r in results) / len(results),
                "avg_improvement": sum(r.improvement for r in results) / len(results),
                "avg_time_semantic": sum(r.time_semantic for r in results) / len(results),
                "avg_time_reranked": sum(r.time_reranked for r in results) / len(results)
            },
            "detailed_results": []
        }
        
        for result in results:
            output_data["detailed_results"].append({
                "question": result.test_case.question,
                "category": result.test_case.category,
                "difficulty": result.test_case.difficulty,
                "expected_answers": result.test_case.expected_answers,
                "expected_videos": result.test_case.expected_videos,
                "semantic_score": result.semantic_score,
                "reranked_score": result.reranked_score,
                "improvement": result.improvement,
                "time_semantic": result.time_semantic,
                "time_reranked": result.time_reranked,
                "semantic_top_results": [
                    {
                        "video": r["video_file"],
                        "timestamp": r["timestamp_display"],
                        "score": r["relevance_score"],
                        "text": r["text"][:100] + "..."
                    } for r in result.semantic_only_results[:3]
                ],
                "reranked_top_results": [
                    {
                        "video": r["video_file"],
                        "timestamp": r["timestamp_display"],
                        "score": r["relevance_score"],
                        "rerank_score": r.get("rerank_score"),
                        "text": r["text"][:100] + "..."
                    } for r in result.reranked_results[:3]
                ]
            })
        
        filename = "rag_evaluation_results.json"
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {filename}")

def run_specific_test(question: str):
    """Run evaluation for a specific question"""
    evaluator = RAGEvaluator()
    evaluator.setup_index()
    
    # Create a custom test case
    test_case = TestCase(
        question=question,
        expected_answers=[],  # No predefined answers for custom questions
        expected_videos=[],
        category="Custom",
        difficulty="unknown",
        description="User-provided question"
    )
    
    result = evaluator.evaluate_test_case(test_case)
    
    print(f"\n📋 DETAILED RESULTS:")
    print(f"Question: {question}")
    print(f"\n🔍 Semantic Only Results:")
    for i, res in enumerate(result.semantic_only_results[:3], 1):
        print(f"  {i}. {res['video_file']} | {res['timestamp_display']} | Score: {res['relevance_score']:.3f}")
        print(f"     {res['text'][:80]}...")
    
    print(f"\n🚀 With Reranker Results:")
    for i, res in enumerate(result.reranked_results[:3], 1):
        rerank_info = f" | Rerank: {res.get('rerank_score', 'N/A'):.3f}" if res.get('rerank_score') else ""
        print(f"  {i}. {res['video_file']} | {res['timestamp_display']} | Score: {res['relevance_score']:.3f}{rerank_info}")
        print(f"     {res['text'][:80]}...")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test specific question from command line
        question = " ".join(sys.argv[1:])
        run_specific_test(question)
    else:
        # Run full evaluation suite
        evaluator = RAGEvaluator()
        evaluator.run_evaluation()
        print(f"\n✅ Evaluation complete! Check rag_evaluation_results.json for detailed results.") 