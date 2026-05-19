import operator
import os
from typing import TypedDict, List, Dict, Annotated, Literal, Any
from pydantic import BaseModel, Field

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# -----------------------------------------------------
# 🗂 STATE DEFINITION
# -----------------------------------------------------
class GraphState(TypedDict):
    user_query: str
    pages: List[str]
    current_page_index: int
    extracted_data: Dict[int, dict]
    cross_page_insights: Annotated[List[str], operator.add]
    partial_answer: str
    final_answer: str
    validation_status: bool
    iteration_count: int
    max_iterations: int
    reasoning_trace: Annotated[List[str], operator.add]
    page_analysis: str  # Transient payload storage

# Initialize LLMs (Using Mistral-Large for complex JSON validation and routing logic)
# Ensure the MISTRAL_API_KEY environment variable is set
llm = ChatMistralAI(model="mistral-large-latest", temperature=0)

# -----------------------------------------------------
# 🧠 NODES
# -----------------------------------------------------

def page_analyzer(state: GraphState):
    idx = state["current_page_index"]
    pages = state["pages"]
    
    # Safe guard
    page_content = pages[idx] if idx < len(pages) else "No more pages left."
    query = state["user_query"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",
                "You are a page-grounded analyzer. "
                "You MUST ONLY use information explicitly written in the provided page content. "
                "Do NOT add external knowledge. "
                "Do NOT infer facts not directly stated. "
                "If something is not present, do not fabricate it. "
                "Output a detailed semantic summary strictly grounded in the page text."),
        ("user", "Query: {query}\n\nPage Content:\n{page}")
    ])
    
    response = llm.invoke(prompt.format_messages(query=query, page=page_content))
    
    return {
        "page_analysis": response.content,
        "reasoning_trace": [f"Page Analyzer: Initially assessed page {idx}."]
    }

class ExtractedData(BaseModel):
    entities: List[str] = Field(description="Exact entities (names, places, etc.)")
    key_claims: List[str] = Field(description="Key claims found on the page")
    metrics: List[str] = Field(description="Numerical metrics, percentages, amounts")
    dates: List[str] = Field(description="Dates or timestamps")
    definitions: List[str] = Field(description="Explicit definitions provided")
    confidence_score: float = Field(description="Confidence (0.0 to 1.0) that data is relevant to query")

def structured_data_extraction(state: GraphState):
    idx = state["current_page_index"]
    analysis = state["page_analysis"]
    
    extractor = llm.with_structured_output(ExtractedData)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract detailed structured data from the provided page analysis. "
                   "If a field has no data, return an empty list."),
        ("user", "Page Analysis:\n{analysis}")
    ])
    
    res = extractor.invoke(prompt.format_messages(analysis=analysis))
    
    # Merge safely into global dictionary state
    current_data = state.get("extracted_data", {})
    current_data[idx] = res.model_dump()
    
    return {
        "extracted_data": current_data,
        "reasoning_trace": [f"Data Extraction: Mapped structured entities for page {idx}."]
    }

class ReasoningOutput(BaseModel):
    new_insights: List[str] = Field(description="New synthesized facts or contradictions found across pages")
    updated_partial_answer: str = Field(description="The evolving answer unifying all parts of the query so far")

def cross_page_reasoning(state: GraphState):
    query = state["user_query"]
    partial = state.get("partial_answer", "")
    all_data = state.get("extracted_data", {})
    insights = state.get("cross_page_insights", [])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",
            "You are a strictly grounded Cross-Page Reasoner. "
            "You MUST ONLY use the extracted structured data provided. "
            "You are NOT allowed to use any outside knowledge about the company. "
            "If information is missing from the extracted data, do not guess. "
            "Only synthesize relationships between the provided data."),
        ("user", "Query: {query}\nCurrent Partial Answer: {partial}\nAll Data: {data}\nPast Insights: {insights}")
    ])
    
    reasoner = llm.with_structured_output(ReasoningOutput)
    res = reasoner.invoke(prompt.format_messages(query=query, partial=partial, data=all_data, insights=insights))
    
    return {
        "cross_page_insights": res.new_insights if res and hasattr(res, "new_insights") and res.new_insights is not None else [],
        "partial_answer": res.updated_partial_answer if res and hasattr(res, "updated_partial_answer") and res.updated_partial_answer is not None else partial,
        "reasoning_trace": ["Cross Page Reasoning: Synthesized insights and updated partial answer."]
    }

def advance_page_node(state: GraphState):
    """Safely increments page tracking prior to conditional loop cycles."""
    return {
        "current_page_index": state["current_page_index"] + 1,
        "iteration_count": state["iteration_count"] + 1,
        "reasoning_trace": [f"System: Proceeding to page {state['current_page_index'] + 1}, iteration {state['iteration_count'] + 1}"]
    }

def query_resolution(state: GraphState):
    query = state["user_query"]
    partial = state.get("partial_answer", "")
    all_data = state.get("extracted_data", {})
    
    prompt = ChatPromptTemplate.from_messages([
       ("system",
            "You are the Final Resolver. "
            "You MUST generate the final answer using ONLY the extracted data dictionary. "
            "Do NOT introduce any new facts, metrics, numbers, or entities "
            "that are not present in the extracted data. "
            "Every factual statement must correspond to a page index in the extracted data. "
            "If data is insufficient, state that explicitly instead of guessing."),
        ("user", "Query: {query}\nPartial Answer: {partial}\nExtracted Data: {data}")
    ])
    
    res = llm.invoke(prompt.format_messages(query=query, partial=partial, data=all_data))
    
    return {
        "final_answer": res.content,
        "reasoning_trace": ["Query Resolution: Formatted final answer with page citations."]
    }

class ValidationResult(BaseModel):
    is_valid: bool = Field(description="Is the answer fully grounded, accurate, and completely addressing the query without contradiction?")
    missing_aspect: str = Field(description="Describe what is missing or broken. Put 'None' if valid.")

def validation(state: GraphState):
    query = state["user_query"]
    final = state.get("final_answer", "")
    data = state.get("extracted_data", {})
    
    validator = llm.with_structured_output(ValidationResult)
    prompt = ChatPromptTemplate.from_messages([
       ("system",
            "You are a strict grounding validator. "
            "Check that every claim in the final answer exists in the extracted data. "
            "If any metric, number, entity, or claim appears that is not in the extracted data, "
            "mark is_valid = False and explain what was hallucinated."),
        ("user", "Query: {query}\nFinal Answer: {final}\nData: {data}")
    ])
    
    res = validator.invoke(prompt.format_messages(query=query, final=final, data=data))
    
    return {
        "validation_status": res.is_valid,
        "reasoning_trace": [f"Validation: Result={res.is_valid}, Notes={res.missing_aspect}"]
    }

# -----------------------------------------------------
# 🧭 CONDITIONAL EDGES (ROUTERS)
# -----------------------------------------------------

def sufficiency_evaluation(state: GraphState) -> Literal["analyze_next_page", "resolve_answer", "force_terminate"]:
    query = state["user_query"]
    partial = state.get("partial_answer", "")
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 5)
    idx = state.get("current_page_index", 0)
    total_pages = len(state.get("pages", []))
    
    # Immediate forced termination if we exceed iterations to prevent infinite loops
    if iteration >= max_iter:
        return "force_terminate"
        
    # Guard to prevent index out of bounds before LLM check to save tokens and prevent looping on the last page forever
    if idx >= total_pages - 1:
        return "force_terminate"
    
    class SufficiencyCheck(BaseModel):
        is_complete: bool = Field(description="Does the partial answer fully cover the original query with no missing aspects?")
        
    checker = llm.with_structured_output(SufficiencyCheck)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Determine if the partial answer completely satisfies the query."),
        ("user", "Query: {query}\nPartial Answer: {partial}")
    ])
    
    check_result = checker.invoke(prompt.format_messages(query=query, partial=partial))
    
    # Check completeness
    if check_result and hasattr(check_result, "is_complete") and check_result.is_complete:
        return "resolve_answer"
        
    return "analyze_next_page"

def post_validation_router(state: GraphState) -> Literal["advance_page_node", "__end__"]:
    # If the response was fully valid, end the graph
    if state.get("validation_status", False) is True:
        return "__end__"
        
    # If we failed validation but reached maximum iteration/pages, we must terminate
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 5)
    idx = state.get("current_page_index", 0)
    total_pages = len(state.get("pages", []))
    
    if iteration >= max_iter or idx >= total_pages - 1:
        return "__end__"
        
    # Validation failed and we have room to search deeper (should rarely hit this due to sufficiency router guards)
    return "advance_page_node"

# -----------------------------------------------------
# ⛓ GRAPH BUILD & COMPILE
# -----------------------------------------------------

workflow = StateGraph(GraphState)

workflow.add_node("page_analyzer", page_analyzer)
workflow.add_node("data_extraction", structured_data_extraction)
workflow.add_node("cross_page_reasoning", cross_page_reasoning)
workflow.add_node("advance_page_node", advance_page_node)
workflow.add_node("query_resolution", query_resolution)
workflow.add_node("validation", validation)

# Baseline Flow
workflow.add_edge(START, "page_analyzer")
workflow.add_edge("page_analyzer", "data_extraction")
workflow.add_edge("data_extraction", "cross_page_reasoning")

# Multi-Agent Iteration Conditional
workflow.add_conditional_edges(
    "cross_page_reasoning",
    sufficiency_evaluation,
    {
        "analyze_next_page": "advance_page_node",
        "resolve_answer": "query_resolution",
        "force_terminate": "query_resolution"
    }
)

# Iteration Loop Back
workflow.add_edge("advance_page_node", "page_analyzer")

# Pre-Flight Output Flow
workflow.add_edge("query_resolution", "validation")

# Post-Validation Conditional
workflow.add_conditional_edges(
    "validation",
    post_validation_router,
    {
        "advance_page_node": "advance_page_node",
        "__end__": END
    }
)

system_graph = workflow.compile()



# -----------------------------------------------------
# 🚀 SAMPLE EXECUTION SCRIPT
# -----------------------------------------------------
# if __name__ == "__main__":
    
#     # Document corpus (5 sequential pages representing document discovery stream)
#     sample_pages = [
#         "Page 0: Reliance Industries Ltd. (RIL) is one of India's largest conglomerates, with diversified operations across energy, petrochemicals, retail, and telecommunications. The company is listed on NSE and BSE and forms a significant weight in benchmark indices like Nifty 50.",
    
#     "Page 1: Equity research indicates that Reliance’s revenue growth is primarily driven by its O2C (Oil-to-Chemicals) segment and strong subscriber additions in Jio Platforms. Analysts closely monitor refining margins, ARPU growth in telecom, and retail expansion as key performance indicators.",
    
#     "Page 2: Financial analysis shows consistent EBITDA growth supported by scale advantages and vertical integration. However, profitability can be sensitive to crude oil price volatility, regulatory changes in telecom, and competitive pressures in the retail segment.",
    
#     "Page 3: Valuation metrics such as P/E ratio, EV/EBITDA, and free cash flow yield suggest that the stock often trades at a premium compared to industry peers, reflecting strong growth expectations and diversified business resilience.",
    
#     "Page 4: Risk factors highlighted in equity reports include global demand slowdown, debt levels from expansion projects, currency fluctuations, and policy risks. Long-term outlook remains positive due to digital ecosystem expansion, green energy investments, and continued market leadership."
#    ]
    
#     initial_state = {
#         "user_query": "How do Reliance Industries Ltd.'s diversified business segments contribute to its premium valuation, and what key risks could impact this growth outlook according to the report?",
#         "pages": sample_pages,
#         "current_page_index": 0,
#         "iteration_count": 1,
#         "max_iterations": 10,
#         "extracted_data": {},
#         "cross_page_insights": [],
#         "reasoning_trace": ["System Initialization"],
#         "partial_answer": "",
#         "final_answer": "",
#         "validation_status": False,
#         "page_analysis": ""
#     }
    
#     print("--- Starting Graph Execution ---\n")
    
#     final_answer = ""
#     validation_status = False
    
#     # We use .stream() instead of .invoke() to watch the graph's thought process step-by-step
#     config = {"recursion_limit": 100}
#     for event in system_graph.stream(initial_state, config=config):
#         for node_name, node_update in event.items():
#             print(f"[ Executed Node: {node_name} ]")
            
#             if "reasoning_trace" in node_update:
#                 for trace in node_update["reasoning_trace"]:
#                     print(f"  -> {trace}")
            
#             if "cross_page_insights" in node_update and node_update["cross_page_insights"]:
#                 print(f"  -> New Insights Found:")
#                 for insight in node_update["cross_page_insights"]:
#                     print(f"     * {insight}")
                    
#             if "final_answer" in node_update:
#                 final_answer = node_update["final_answer"]
                
#             if "validation_status" in node_update:
#                 validation_status = node_update["validation_status"]
            
#             print("-" * 50)
    
#     print("\n========================================")
#     print("Final Answer:")
#     print(final_answer)
    
#     print("\nValidation Passed:", validation_status)



def run_document_graph(pages, user_query, system_graph, max_iterations=10, recursion_limit=100):
    """
    Executes the graph pipeline on a given list of pages and user query.

    Args:
        pages (list[str]): List of document pages.
        user_query (str): User question.
        system_graph: Compiled graph object.
        max_iterations (int): Maximum reasoning iterations.
        recursion_limit (int): Graph recursion limit.

    Returns:
        dict: {
            "final_answer": str,
            "validation_status": bool
        }
    """

    # Initial graph state
    initial_state = {
        "user_query": user_query,
        "pages": pages,
        "current_page_index": 0,
        "iteration_count": 1,
        "max_iterations": max_iterations,
        "extracted_data": {},
        "cross_page_insights": [],
        "reasoning_trace": ["System Initialization"],
        "partial_answer": "",
        "final_answer": "",
        "validation_status": False,
        "page_analysis": ""
    }

    print("--- Starting Graph Execution ---\n")

    final_answer = ""
    validation_status = False

    config = {"recursion_limit": recursion_limit}

    # Stream execution
    for event in system_graph.stream(initial_state, config=config):
        for node_name, node_update in event.items():
            print(f"[ Executed Node: {node_name} ]")

            if "reasoning_trace" in node_update:
                for trace in node_update["reasoning_trace"]:
                    print(f"  -> {trace}")

            if "cross_page_insights" in node_update and node_update["cross_page_insights"]:
                print("  -> New Insights Found:")
                for insight in node_update["cross_page_insights"]:
                    print(f"     * {insight}")

            if "final_answer" in node_update:
                final_answer = node_update["final_answer"]

            if "validation_status" in node_update:
                validation_status = node_update["validation_status"]

            print("-" * 50)

    print("\n========================================")
    print("Final Answer:")
    print(final_answer)
    print("\nValidation Passed:", validation_status)

    return {
        "final_answer": final_answer,
        "validation_status": validation_status
    }