from dataclasses import dataclass, field
from typing import Generic, NewType

import demo_css
import gradio as gr
import utils
from dependency_injector.wiring import Provide, inject
from query_common.entities.conditions.logical import ConjunctiveCondition
from query_common.grounding.actions import ground_query
from query_common.parse.base import Parser  # noqa: TC002

from complex_query.core.contracts import KnowledgeGraph, LinkPredictor, ScoreLedger  # noqa: TC001
from complex_query.core.entities.config.score_aggregator import ProductScoreAggregatorConfig
from complex_query.core.values.errors import SearchInputError
from complex_query.impl.score_aggregator.product import ProductScoreAggregator
from complex_query.impl.search.beam_search import (
    QueryConditionStep,
    QueryPartialSolution,
    QuerySimulator,
)
from complex_query.impl.search.beam_search.generic import (
    BeamNode,
    BeamSearch,
    BeamStep,
    PartialSolutionT,
    StepT,
)

NodeID = NewType("NodeID", int)


@dataclass
class TreeNode(Generic[PartialSolutionT, StepT]):
    node: BeamNode[PartialSolutionT, StepT]
    children: dict[NodeID, "TreeNode[PartialSolutionT, StepT]"] = field(default_factory=dict)
    steps_chosen: dict[NodeID, StepT] = field(default_factory=dict)
    beam_step_options: list[BeamStep[PartialSolutionT, StepT]] = field(default_factory=list)


def reconstruct_beam_path(
    final_nodes: list[BeamNode[PartialSolutionT, StepT]],
) -> list[BeamNode[PartialSolutionT, StepT]]:
    """
    Given a list of final BeamNodes, reconstruct the full path of visited BeamNodes.

    Args:
    final_nodes (List[BeamNode]): A list of the final nodes from the beam search.

    Returns:
    List[BeamNode]: A list of all visited BeamNodes in order, starting from the initial node.
    """
    all_nodes_ids = set()
    all_nodes = []
    to_process = final_nodes.copy()

    # Collect all nodes
    while to_process:
        node = to_process.pop()
        if id(node) not in all_nodes_ids:
            all_nodes_ids.add(id(node))
            all_nodes.append(node)
            if node.from_beam_step:
                to_process.append(node.from_beam_step.from_beam_node)

    # Sort nodes based on their depth in the tree
    return sorted(all_nodes, key=lambda node: len(node.get_score_path()))


def build_tree_structure(
    beam_nodes: list[BeamNode[PartialSolutionT, StepT]],
) -> dict[NodeID, TreeNode[PartialSolutionT, StepT]]:
    all_tree_nodes: dict[NodeID, TreeNode[PartialSolutionT, StepT]] = {}
    root_nodes: dict[NodeID, TreeNode[PartialSolutionT, StepT]] = {}

    def get_or_create_node(
        node_id: NodeID, node: BeamNode[PartialSolutionT, StepT]
    ) -> TreeNode[PartialSolutionT, StepT]:
        if node_id not in all_tree_nodes:
            all_tree_nodes[node_id] = TreeNode(
                node=node, beam_step_options=node.beam_step_options or []
            )
        return all_tree_nodes[node_id]

    for end_node in beam_nodes:
        step_path: list[BeamStep[PartialSolutionT, StepT]] = []
        node_path: list[BeamNode[PartialSolutionT, StepT]] = [end_node]

        # Build paths of steps and nodes
        current_node = end_node
        while current_node.from_beam_step:
            step_path.append(current_node.from_beam_step)
            current_node = current_node.from_beam_step.from_beam_node
            node_path.append(current_node)
        root_nodes[NodeID(id(current_node))] = get_or_create_node(
            NodeID(id(current_node)), current_node
        )

        # Reverse paths to start from the root
        step_path.reverse()
        node_path.reverse()

        # Build the tree structure
        for i, (node, step) in enumerate(zip(node_path[:-1], step_path, strict=False)):
            current_tree_node = get_or_create_node(NodeID(id(node)), node)
            next_node = node_path[i + 1]
            next_node_id = NodeID(id(next_node))

            current_tree_node.steps_chosen[next_node_id] = step.step
            if next_node_id not in current_tree_node.children:
                current_tree_node.children[next_node_id] = get_or_create_node(
                    next_node_id, next_node
                )

        # Ensure the last node is in the tree
        get_or_create_node(NodeID(id(end_node)), end_node)

    return root_nodes


def generate_tree_html(
    tree: dict[NodeID, TreeNode[PartialSolutionT, StepT]],
) -> str:
    def node_to_html(node_data: TreeNode[PartialSolutionT, StepT]) -> str:
        node = node_data.node
        html = f'<li><a href="#" class="{"end-of-text" if node.is_final else "nonfinal"}">'
        html += (
            f"<span><b>{node.partial_solution.short_repr()}</b>"
            f"<br>Score: {node.cumulative_score:.2f}</span>"
        )

        if node_data.beam_step_options:
            html += "<table><tr><th>Step</th><th>Step score</th><th>Total score</th></tr>"
            for step in node_data.beam_step_options:
                is_selected = any(
                    chosen_step == step.step for chosen_step in node_data.steps_chosen.values()
                )
                row_class = "chosen-step" if is_selected else ""
                html += (
                    f'<tr class="{row_class}"><td>{step.step.short_repr()}</td>'
                    f"<td>{step.score:.4f}</td><td>{step.cumulative_score:.4f}</td></tr>"
                )
            html += "</table>"

        html += "</a>"

        if node_data.children:
            html += "<ul>"
            for _child_id, child_node_data in node_data.children.items():
                html += node_to_html(child_node_data)
            html += "</ul>"

        html += "</li>"
        return html

    tree_html = '<div class="custom-container"><div class="tree"><ul id="root">'
    for _root_id, root_data in tree.items():
        tree_html += node_to_html(root_data)
    tree_html += "</ul></div></div>"
    return tree_html


def beam_search_visualizer(
    final_nodes: list[BeamNode[PartialSolutionT, StepT]],
) -> tuple[str, str]:
    all_nodes = reconstruct_beam_path(final_nodes)
    tree = build_tree_structure(all_nodes)
    tree_html = generate_tree_html(tree)

    markdown = "## Beam Search Results\n\n"
    for i, node in enumerate(all_nodes):
        if node.is_final:
            markdown += f"### Beam {i}\n"
            markdown += f"- Final solution: `{node.partial_solution.short_repr()}`\n"
            markdown += f"- Final score: `{node.cumulative_score:.2f}`\n"
            markdown += "- Path:\n"
            current_node = node
            while current_node.from_beam_step:
                markdown += (
                    f"  - `{current_node.from_beam_step.step.short_repr()}: "
                    f"{current_node.from_beam_step.score:.2f}`\n"
                )
                current_node = current_node.from_beam_step.from_beam_node
            markdown += "\n"

    return tree_html, markdown


@inject
def answer_query_with_visualizer(  # noqa: PLR0913
    query_input: str,
    beam_size: int,
    n_return_candidates: int,
    *args,  # noqa: ARG001, Due to unknown error due to gradio
    parser: Parser = Provide["parser"],
    kg: KnowledgeGraph = Provide["kg"],
    score_ledger: ScoreLedger = Provide["score_ledger"],
    link_predictor: LinkPredictor = Provide["link_predictor"],
) -> tuple[str, str, str]:
    try:
        query = parser.parse_query(query_input)
        grounded_query = ground_query(query)

        if not isinstance(grounded_query.condition, ConjunctiveCondition):
            raise SearchInputError("Gradio demo currently does not support disjunctive queries.")

        conjunctive_conditions = grounded_query.condition.flattened_conditions()
        query_simulator = QuerySimulator(
            kg=kg,
            ledger=score_ledger,
            link_predictor=link_predictor,
            conjunctive_conditions=conjunctive_conditions,
        )
        beam_search = BeamSearch[QuerySimulator, QueryPartialSolution, QueryConditionStep](
            problem_simulator=query_simulator,
            beam_size=beam_size,
            stop_at_first_final_solution=False,
            score_aggregator=ProductScoreAggregator(ProductScoreAggregatorConfig()),
        )
        initial_solution = QueryPartialSolution.from_empty()
        final_nodes = beam_search.search(initial_solution)
        final_nodes = sorted(final_nodes, key=lambda x: x.cumulative_score, reverse=True)
        final_nodes = final_nodes[:n_return_candidates]
        tree_html, markdown = beam_search_visualizer(final_nodes)
        error_output = ""
    except Exception as e:
        tree_html = ""
        markdown = ""
        error_output = (
            f"<div style='color: red; padding: 10px; border: 1px solid red; border-radius: 5px;'>"
            f"Error: {e!s}</div>"
        )
    return tree_html, markdown, error_output


def change_num_return_candidates(beam_size):
    return gr.Slider(
        label="Number of return candidates",
        minimum=1,
        maximum=beam_size,
        step=1,
        value=beam_size,
    )


# Gradio interface
with gr.Blocks(css=demo_css.STYLE, theme=gr.themes.Default()) as demo:
    gr.Markdown("# Beam Search Visualizer")
    gr.Markdown("This app visualizes the output of a beam search algorithm.")

    query_input = gr.Textbox(
        label="Enter your query here",
        value=(
            "? x in DRUG_RELATED_CONCEPT, b in '201000010099'"
            " WHERE P(x, '232000000216', b) AND P(x, '232000000087', '241001148079')"
        ),
    )
    with gr.Row():
        beam_size = gr.Slider(label="Number of beams", minimum=1, maximum=10, step=1, value=4)
        n_return_candidates = gr.Slider(
            label="Number of return candidates", minimum=1, maximum=4, step=1, value=4
        )
    beam_size.change(fn=change_num_return_candidates, inputs=beam_size, outputs=n_return_candidates)
    run_button = gr.Button("Run Beam Search")
    error_output = gr.HTML()
    tree_html = gr.HTML()
    output_markdown = gr.Markdown()
    run_button.click(
        answer_query_with_visualizer,
        inputs=[query_input, beam_size, n_return_candidates],
        outputs=[tree_html, output_markdown, error_output],
    )

if __name__ == "__main__":
    utils.wire_container(__name__)
    demo.launch()
