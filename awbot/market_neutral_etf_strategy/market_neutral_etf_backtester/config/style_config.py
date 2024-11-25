import plotly.graph_objects as go


MODERN_COLORS = {
    "background_color": "#000000",  # Pure black
    "paper_color": "#121212",  # Very dark gray
    "text_color": "#FFFFFF",  # Pure white
    "grid_color": "#333333",  # Dark gray
    "accent_colors": [
        "#00A3FF",  # Vibrant blue
        "#00FF85",  # Electric green
        "#FF00E5",  # Magenta
        "#FF9900",  # Orange
        "#FFD600",  # Yellow
        "#00E5FF",  # Cyan
        "#FF4081",  # Pink
    ],
}


def create_modern_theme():
    """Create a high-contrast theme optimized for screen recording."""
    modern = go.layout.Template()

    # Set global defaults
    modern.layout = dict(
        font=dict(
            family="Arial, sans-serif",  # Changed to Arial for better screen legibility
            color=MODERN_COLORS["text_color"],
            size=14,  # Increased font size
        ),
        paper_bgcolor=MODERN_COLORS["paper_color"],
        plot_bgcolor=MODERN_COLORS["background_color"],
        title=dict(font=dict(size=20, color=MODERN_COLORS["text_color"]), x=0.5, xanchor="center"),
        # Grid styling
        xaxis=dict(
            gridcolor=MODERN_COLORS["grid_color"],
            linecolor=MODERN_COLORS["grid_color"],
            zerolinecolor=MODERN_COLORS["grid_color"],
            tickcolor=MODERN_COLORS["text_color"],
            tickfont=dict(color=MODERN_COLORS["text_color"], size=12),
            title=dict(font=dict(color=MODERN_COLORS["text_color"], size=14)),
        ),
        yaxis=dict(
            gridcolor=MODERN_COLORS["grid_color"],
            linecolor=MODERN_COLORS["grid_color"],
            zerolinecolor=MODERN_COLORS["grid_color"],
            tickcolor=MODERN_COLORS["text_color"],
            tickfont=dict(color=MODERN_COLORS["text_color"], size=12),
            title=dict(font=dict(color=MODERN_COLORS["text_color"], size=14)),
        ),
        legend=dict(
            bgcolor=MODERN_COLORS["paper_color"],
            font=dict(color=MODERN_COLORS["text_color"], size=12),
            bordercolor=MODERN_COLORS["grid_color"],
            borderwidth=1,
        ),
        colorway=MODERN_COLORS["accent_colors"],
    )

    # Scatter plot defaults with increased visibility
    modern.data.scatter = [
        go.Scatter(line=dict(width=3), marker=dict(size=10))  # Thicker lines  # Larger markers
    ]

    # Box plot defaults
    modern.data.box = [
        go.Box(
            marker=dict(
                outliercolor=MODERN_COLORS["accent_colors"][2],  # Magenta for outliers
                color=MODERN_COLORS["accent_colors"][0],  # Blue for boxes
                size=8,  # Larger markers
            ),
            line=dict(color=MODERN_COLORS["text_color"], width=2),
            fillcolor=MODERN_COLORS["accent_colors"][0],
        )
    ]

    # Bar plot defaults
    modern.data.bar = [go.Bar(marker=dict(line=dict(color=MODERN_COLORS["text_color"], width=1)))]

    return modern
