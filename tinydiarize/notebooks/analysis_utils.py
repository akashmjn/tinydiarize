import glob
import re
from pathlib import Path

import IPython.display as ipd
import matplotlib.pyplot as plt
import pandas as pd
from score import parse_analysis_file

# edit matplotlib theme to a better looking one
plt.style.use("tableau-colorblind10")


def parse_result_id(input):
    """Parse the result_id into model, call_id, method"""
    # rename some methods to more understandable names
    method_map = {
        "drz_pre_sr__segment": "pyannote_pre_sr",
        "segment": "segment_time",
        "drz_post_sr__segment": "segment_time_clustered",
        "token": "tdrz_token",
    }

    def _parse_result_id(result_id):
        model, call_id, method = re.search(
            r"(.*)__earnings21-([0-9]+)_+(.*)", result_id
        ).groups()
        model = model.split("_")[0]  # remove the suffix after -ft_*
        method = method_map.get(method, method)
        return model, call_id, method

    if isinstance(input, str):
        return _parse_result_id(input)
    else:
        return zip(*[_parse_result_id(r) for r in input])


def compile_results(results_dir):
    # read and compile both scoring_results and analysis_results
    # compile all scoring_results.tsv files
    scored_tsvs = []
    tsv_list = glob.glob(f"{results_dir}/**/scoring_results.tsv", recursive=True)
    for tsv in tsv_list:
        scored_tsvs.append(pd.read_csv(tsv, sep="\t"))
    results_df = pd.concat(scored_tsvs)
    results_df["model"], results_df["call_id"], results_df["method"] = parse_result_id(
        results_df.result_id
    )
    print(f"Read {len(results_df)} results from {len(tsv_list)} files")

    # # collect all side-by-side analysis results
    analysis_results = dict()
    analysis_sbs_list = glob.glob(
        f"{results_dir}/**/spk_turn/results/*.sbs", recursive=True
    )
    for sbs in analysis_sbs_list:
        # get the result_id from the path
        result_id = Path(sbs).parts[-4]  # directory name of **
        precision_errors, recall_errors = parse_analysis_file(sbs)
        key_tuple = parse_result_id(result_id)
        analysis_results[key_tuple] = dict(
            precision_errors=precision_errors, recall_errors=recall_errors
        )
    print(f"Read {len(analysis_results)} side-by-side analysis results")

    return results_df, analysis_results


def query_metric_results(results_df, metric, groups=["call_id", "method"]):
    """Query the results for a given metric"""
    metric_df = (
        results_df.query(f'metric=="{metric}"')
        .groupby(groups)["value"]
        .first()
        .unstack()
        .round(2)
    )
    return metric_df


def plot_metric_results(metric_df, title=None, ax=None, legend=True):
    if ax is None:
        ax = plt.gca()
    metric_df.plot.barh(title=title, ax=ax, legend=legend, grid=True)
    if legend:
        # edit the legend of the plot so that it doesn't overlap with the plot
        _ = ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)


def inspect_errors(
    results_df, analysis_results, result_key, precision_errors=[], recall_errors=[]
):
    model_id, call_id, method = result_key

    print(f"Results for: {result_key}")
    ipd.display(
        results_df.query(
            f'model=="{model_id}" and call_id=="{call_id}" and method=="{method}"'
        ).iloc[:, :7]
    )

    spk_turn_errors = analysis_results[result_key]
    print(f"Precision errors: {len(spk_turn_errors['precision_errors'])}")
    print(f"Recall errors: {len(spk_turn_errors['recall_errors'])}")

    if len(precision_errors) > 0:
        print("\n", "--" * 5, "Precision errors:", "--" * 5)
        for idx in precision_errors:
            print("\nLine:", spk_turn_errors["precision_errors"][idx]["line"])
            print(spk_turn_errors["precision_errors"][idx]["context"])

    if len(recall_errors) > 0:
        print("\n", "--" * 5, "Recall errors:", "--" * 5)
        for idx in recall_errors:
            print("\nLine:", spk_turn_errors["recall_errors"][idx]["line"])
            print(spk_turn_errors["recall_errors"][idx]["context"])


"""
Nice-to-have TODOs:
- parse analysis results into custom class with
    errors uniquely identified by ref word #
    configurable context
- enable diff between two sets of errors
- make a neat side-by-side fixed width print
"""


# function to print two strings side-by-side with a fixed width
def print_side_by_side(s1, s2, width=50):
    # split the strings into lines
    s1_lines = s1.splitlines()
    s2_lines = s2.splitlines()
    # get the maximum number of lines
    max_lines = max(len(s1_lines), len(s2_lines))
    # pad the lines with empty strings
    s1_lines += [""] * (max_lines - len(s1_lines))
    s2_lines += [""] * (max_lines - len(s2_lines))
    # print the lines side-by-side
    for s1, s2 in zip(s1_lines, s2_lines):
        s1 = s1.rsplit("\t", 2)[
            0
        ]  # remove the last 2 columns, keep only words and ERR hints
        s2 = s2.rsplit("\t", 2)[0]
        print(f"{s1: <{width}}{s2: <{width}}")
