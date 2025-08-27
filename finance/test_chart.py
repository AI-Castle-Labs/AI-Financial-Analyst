# Added robust import with fallback
try:
    import vizro.plotly.express as px  # type: ignore
    from vizro.models.types import capture  # type: ignore
except ModuleNotFoundError:  # Fallback if vizro isn't installed
    import plotly.express as px  # type: ignore
    def capture(_):  # no-op decorator when vizro absent
        def _wrap(func):
            return func
        return _wrap

import pandas as pd
from typing import Any

@capture("graph")
def custom_chart(data_frame: pd.DataFrame) -> Any:
    """Return a horizontal bar chart of the top 20 most common positions.

    Handles cases where the Position column may be missing or empty.
    Adds defensive normalization of column names to avoid ValueError where
    plotly complains that 'Position' not found due to unexpected renaming.
    """
    if "Position" not in data_frame.columns:
        raise KeyError("'Position' column not found in provided DataFrame")

    # Compute counts using more explicit reset_index(name=...) to avoid ambiguity
    position_counts = (
        data_frame["Position"].dropna().astype(str).value_counts().head(20).reset_index(name="Frequency")
    )
    # After reset_index, the default column holding the category labels is 'index'. Rename to 'Position'.
    if "index" in position_counts.columns and "Position" not in position_counts.columns:
        position_counts = position_counts.rename(columns={"index": "Position"})

    # Defensive fallback heuristics (in case upstream code altered structure)
    if "Position" not in position_counts.columns:
        # Try to infer: the non-frequency column with object dtype & many unique values
        for col in position_counts.columns:
            if col.lower() in ("role", "title"):
                position_counts = position_counts.rename(columns={col: "Position"})
                break
        if "Position" not in position_counts.columns:
            # Last resort: take the first column as Position
            position_counts = position_counts.rename(columns={position_counts.columns[0]: "Position"})

    if "Frequency" not in position_counts.columns:
        # Common alternative names
        for alt in ("count", "counts", "value", "freq"):
            if alt in position_counts.columns:
                position_counts = position_counts.rename(columns={alt: "Frequency"})
                break
        if "Frequency" not in position_counts.columns and len(position_counts.columns) > 1:
            # Assume second column is counts
            position_counts = position_counts.rename(columns={position_counts.columns[1]: "Frequency"})

    # Final validation
    expected_cols = {"Position", "Frequency"}
    if not expected_cols.issubset(position_counts.columns):
        raise ValueError(f"Could not normalize columns to {expected_cols}. Got: {list(position_counts.columns)}")

    if position_counts.empty:
        raise ValueError("No data available to plot after processing 'Position' column.")

    # Build figure (horizontal bar)
    fig = px.bar(
        position_counts,
        x="Frequency",
        y="Position",
        orientation="h",
        title="Top 20 Most Common Positions",
        text="Frequency",
    )

    # Improve layout & readability
    fig.update_layout(
        yaxis_title="Position",
        xaxis_title="Frequency",
        yaxis=dict(categoryorder="total ascending"),
        margin=dict(l=80, r=40, t=60, b=40),
    )
    fig.update_traces(textposition="outside", cliponaxis=False)

    return fig

# Example usage / manual test guarded to avoid side-effects on import
if __name__ == "__main__":
    sample_data = {
        "Position": [
            "Software Engineer", "Data Scientist", "Product Manager",
            "Software Engineer", "Data Analyst", "Data Scientist",
            "Software Engineer", "Consultant", "Consultant",
            "Data Engineer", "Data Scientist", "Product Manager",
            "Consultant", "Software Engineer", "Data Analyst",
            "Data Engineer", "Product Manager", "Consultant",
            "Data Scientist", "Software Engineer", "Software Engineer"
        ]
    }
    df = pd.DataFrame(sample_data)
    fig = custom_chart(df)
    try:
        fig.show()
    except Exception as e:
        print("Unable to render interactive chart (possibly no display). Saving to HTML instead.")
        fig.write_html("custom_chart_debug.html")
        print("Chart written to custom_chart_debug.html", e)