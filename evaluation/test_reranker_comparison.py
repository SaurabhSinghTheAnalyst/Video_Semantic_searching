#!/usr/bin/env python3
"""
Comprehensive Reranker Comparison Test

This script compares search results with and without reranking to demonstrate
the improvements provided by cross-encoder reranking.
"""

import os
import json
import time
from typing import List, Dict, Any
from pathlib import Path

from video_index_manager import VideoIndexManager
from video_search_engine import VideoSearchEngine

def format_result_summary(result: Dict[str, Any]) -> str:
    """Format a search result for display"""
    return (f"{result['video_file']} | "
            f"Score: {result.get('relevance_score', 0):.3f} | "
            f"Time: {result['timestamp_display']} | "
            f"Text: {result['text'][:50]}...")

def compare_search_methods(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Compare search results with and without reranker"""
    
    print(f"\n🔍 Query: '{query}'")
    print("=" * 70)
    
    # Load index
    manager = VideoIndexManager()
    index = manager.get_or_create_index('transcripts')
    
    # Test without reranker
    print("\n📊 WITHOUT Reranker (Semantic Search Only):")
    search_engine_no_rerank = VideoSearchEngine(index, use_reranker=False)
    
    start_time = time.time()
    results_no_rerank = search_engine_no_rerank.semantic_search(query, top_k=top_k)
    time_no_rerank = time.time() - start_time
    
    for i, result in enumerate(results_no_rerank, 1):
        print(f"  {i}. {format_result_summary(result)}")
    
    # Test with reranker
    print(f"\n🚀 WITH Reranker:")
    search_engine_rerank = VideoSearchEngine(index, use_reranker=True)
    
    start_time = time.time()
    results_rerank = search_engine_rerank.semantic_search(query, top_k=top_k)
    time_rerank = time.time() - start_time
    
    for i, result in enumerate(results_rerank, 1):
        method = result.get('search_method', 'unknown')
        rerank_score = result.get('rerank_score')
        orig_score = result.get('original_semantic_score', result.get('relevance_score', 0))
        
        score_info = f"Rerank: {rerank_score:.3f}, Orig: {orig_score:.3f}" if rerank_score is not None else f"Score: {orig_score:.3f}"
        
        print(f"  {i}. {result['video_file']} | {score_info} | "
              f"Time: {result['timestamp_display']} | Text: {result['text'][:50]}...")
    
    # Performance comparison
    print(f"\n⏱️  Performance:")
    print(f"  Without reranker: {time_no_rerank:.2f}s")
    print(f"  With reranker: {time_rerank:.2f}s")
    print(f"  Overhead: {time_rerank - time_no_rerank:.2f}s")
    
    # Analyze differences
    print(f"\n📈 Analysis:")
    
    # Check if top results changed
    top_3_no_rerank = [r['video_file'] + '_' + r['timestamp_display'] for r in results_no_rerank[:3]]
    top_3_rerank = [r['video_file'] + '_' + r['timestamp_display'] for r in results_rerank[:3]]
    
    if top_3_no_rerank != top_3_rerank:
        print("  ✅ Reranker changed the top 3 results order")
    else:
        print("  ➡️  Top 3 results order remained the same")
    
    # Check score improvements
    reranked_results = [r for r in results_rerank if r.get('search_method') == 'reranked']
    if reranked_results:
        print(f"  📊 {len(reranked_results)} results were reranked")
        
        # Find the most improved result
        best_improvement = 0
        best_result = None
        for result in reranked_results:
            rerank_score = result.get('rerank_score', 0)
            orig_score = result.get('original_semantic_score', 0)
            improvement = rerank_score - orig_score
            if improvement > best_improvement:
                best_improvement = improvement
                best_result = result
        
        if best_result and best_improvement > 0:
            print(f"  🎯 Best improvement: {best_improvement:.3f} for {best_result['video_file']} at {best_result['timestamp_display']}")
    
    return {
        'query': query,
        'results_no_rerank': results_no_rerank,
        'results_rerank': results_rerank,
        'time_no_rerank': time_no_rerank,
        'time_rerank': time_rerank,
        'top_changed': top_3_no_rerank != top_3_rerank
    }

def run_comprehensive_test():
    """Run comprehensive reranker comparison tests"""
    
    print("🚀 COMPREHENSIVE RERANKER COMPARISON TEST")
    print("=" * 60)
    
    # Test queries with different characteristics
    test_queries = [
        "contextual retrieval RAG systems",
        "machine learning algorithms",
        "building neural networks",
        "data preprocessing techniques",
        "python programming best practices",
        "artificial intelligence applications"
    ]
    
    results_summary = []
    
    for query in test_queries:
        try:
            result = compare_search_methods(query, top_k=5)
            results_summary.append(result)
            
            # Add separator between queries
            print("\n" + "─" * 70)
            
        except Exception as e:
            print(f"❌ Error testing query '{query}': {e}")
    
    # Overall summary
    print(f"\n🎯 OVERALL SUMMARY")
    print("=" * 40)
    
    total_queries = len(results_summary)
    queries_with_changes = sum(1 for r in results_summary if r['top_changed'])
    avg_time_no_rerank = sum(r['time_no_rerank'] for r in results_summary) / total_queries
    avg_time_rerank = sum(r['time_rerank'] for r in results_summary) / total_queries
    
    print(f"Total queries tested: {total_queries}")
    print(f"Queries with top-3 changes: {queries_with_changes} ({queries_with_changes/total_queries*100:.1f}%)")
    print(f"Average time without reranker: {avg_time_no_rerank:.2f}s")
    print(f"Average time with reranker: {avg_time_rerank:.2f}s")
    print(f"Average reranking overhead: {avg_time_rerank - avg_time_no_rerank:.2f}s")
    
    print(f"\n✅ Reranker test completed!")
    print(f"💡 The reranker should provide more relevant results for complex queries")
    print(f"🔧 You can now test different queries in the Streamlit app at: http://localhost:8501")

def test_specific_query(query: str = None):
    """Test a specific query interactively"""
    if query is None:
        query = input("\n🔍 Enter your test query: ")
    
    if not query.strip():
        print("❌ Please provide a valid query")
        return
    
    compare_search_methods(query, top_k=5)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test specific query from command line
        query = " ".join(sys.argv[1:])
        test_specific_query(query)
    else:
        # Run comprehensive test
        run_comprehensive_test() 